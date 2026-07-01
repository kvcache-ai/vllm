# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SGLang-style rid-based arrival doorbell.

D-side ``ArrivalDoorbell`` allocates a per-request slot in a GPU buffer
registered with Mooncake; the slot is zeroed and a random nonce is held
in process memory as the expected value. P writes the nonce into the
slot via RDMA as the LAST descriptor of the batch, then ZMQ-replies
``ok_reqs``. D reads its own local slot and verifies the nonce -- a
match proves both the doorbell and all preceding KV descriptors in the
same RDMA batch landed at the remote.

P-side ``NoncePad`` stages a copy of each request's nonce in a small
device-resident scratch buffer (registered with Mooncake) so the
producer can supply the source pointers for those extra descriptors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.arrival_doorbell import (
    ARRIVAL_SLOT_BYTES,
    ArrivalDoorbell,
    NoncePad,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestArrivalDoorbellAllocate:
    def _make(self, capacity: int = 8) -> tuple[MagicMock, ArrivalDoorbell]:
        engine = MagicMock()
        engine.batch_register_memory = MagicMock(return_value=0)
        return engine, ArrivalDoorbell(engine, torch.device("cuda"), capacity=capacity)

    def test_allocate_returns_unique_addrs(self):
        _, db = self._make(capacity=4)
        a = db.allocate("r1")
        b = db.allocate("r2")
        assert a.slot_addr != b.slot_addr
        assert a.expected_nonce != b.expected_nonce

    def test_allocate_zeros_slot_on_device(self):
        _, db = self._make()
        # Pre-poison the buffer.
        db._buffer.fill_(0xAB)
        a = db.allocate("r1")
        slot_offset = a.slot_addr - db._buffer.data_ptr()
        slot = db._buffer[slot_offset : slot_offset + ARRIVAL_SLOT_BYTES]
        assert torch.all(slot == 0).item()

    def test_allocate_raises_when_exhausted(self):
        _, db = self._make(capacity=2)
        db.allocate("r1")
        db.allocate("r2")
        with pytest.raises(RuntimeError, match="exhausted"):
            db.allocate("r3")

    def test_release_lets_capacity_recover(self):
        _, db = self._make(capacity=1)
        a = db.allocate("r1")
        db.release("r1")
        b = db.allocate("r2")
        assert b.slot_addr == a.slot_addr  # reused

    def test_double_release_is_safe(self):
        _, db = self._make()
        db.allocate("r1")
        db.release("r1")
        db.release("r1")  # must not raise

    def test_registers_buffer_with_engine_once(self):
        engine, db = self._make()
        engine.batch_register_memory.assert_called_once()
        ptrs, lens = engine.batch_register_memory.call_args[0]
        assert ptrs == [db._buffer.data_ptr()]
        assert lens == [db._buffer.numel()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestArrivalDoorbellVerify:
    def _make(self):
        engine = MagicMock()
        engine.batch_register_memory = MagicMock(return_value=0)
        return ArrivalDoorbell(engine, torch.device("cuda"), capacity=4)

    def test_verify_matches_after_remote_write(self):
        db = self._make()
        h = db.allocate("r1")
        # Simulate the producer's remote RDMA write landing on the slot.
        slot_offset = h.slot_addr - db._buffer.data_ptr()
        nonce_bytes = h.expected_nonce.to_bytes(8, "little")
        payload = torch.tensor(
            list(nonce_bytes) + [0] * (ARRIVAL_SLOT_BYTES - 8),
            dtype=torch.uint8,
            device=db._buffer.device,
        )
        db._buffer[slot_offset : slot_offset + ARRIVAL_SLOT_BYTES] = payload
        assert db.verify("r1") is True

    def test_verify_fails_when_slot_still_zero(self):
        db = self._make()
        db.allocate("r1")
        # Slot was zeroed at allocation; producer never wrote anything.
        assert db.verify("r1") is False

    def test_verify_fails_on_wrong_nonce(self):
        db = self._make()
        h = db.allocate("r1")
        slot_offset = h.slot_addr - db._buffer.data_ptr()
        bad = (h.expected_nonce ^ 0xDEADBEEF).to_bytes(8, "little")
        payload = torch.tensor(
            list(bad) + [0] * (ARRIVAL_SLOT_BYTES - 8),
            dtype=torch.uint8,
            device=db._buffer.device,
        )
        db._buffer[slot_offset : slot_offset + ARRIVAL_SLOT_BYTES] = payload
        assert db.verify("r1") is False

    def test_verify_unknown_req_returns_false(self):
        db = self._make()
        assert db.verify("never_allocated") is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestNoncePad:
    def _make(self, capacity: int = 8) -> tuple[MagicMock, NoncePad]:
        engine = MagicMock()
        engine.batch_register_memory = MagicMock(return_value=0)
        return engine, NoncePad(engine, torch.device("cuda"), capacity=capacity)

    def test_stage_writes_nonces_and_returns_src_ptrs(self):
        _, pad = self._make(capacity=4)
        nonces = [0x1111111111111111, 0x2222222222222222]
        src_ptrs = pad.stage(nonces)
        assert len(src_ptrs) == 2
        assert src_ptrs[0] != src_ptrs[1]
        # Verify the nonces actually landed on the device.
        for nonce, ptr in zip(nonces, src_ptrs):
            offset = ptr - pad._buffer.data_ptr()
            chunk = pad._buffer[offset : offset + 8].cpu().numpy().tobytes()
            assert int.from_bytes(chunk, "little") == nonce

    def test_stage_round_robins_slots(self):
        _, pad = self._make(capacity=2)
        p1 = pad.stage([1])[0]
        _ = pad.stage([2])[0]
        p3 = pad.stage([3])[0]  # wraps
        assert p3 == p1

    def test_registers_with_engine(self):
        engine, pad = self._make()
        engine.batch_register_memory.assert_called_once()
        ptrs, lens = engine.batch_register_memory.call_args[0]
        assert ptrs == [pad._buffer.data_ptr()]
        assert lens == [pad._buffer.numel()]
