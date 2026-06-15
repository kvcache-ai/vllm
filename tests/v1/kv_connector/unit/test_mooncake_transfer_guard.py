# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer_guard import (
    MooncakeTransferVerifier,
    get_remote_session_lock,
)


class TestMooncakeSessionLock:
    def test_same_session_returns_same_lock(self):
        lock_a = get_remote_session_lock("host:1234")
        lock_b = get_remote_session_lock("host:1234")
        assert lock_a is lock_b

    def test_different_sessions_return_different_locks(self):
        lock_a = get_remote_session_lock("host:1234")
        lock_b = get_remote_session_lock("host:5678")
        assert lock_a is not lock_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMooncakeTransferVerifier:
    def test_verify_succeeds_when_readback_matches(self, monkeypatch):
        monkeypatch.setenv("VLLM_MOONCAKE_TRANSFER_VERIFY_TAIL_BYTES", "16")
        engine = MagicMock()
        engine.batch_register_memory.return_value = 0

        src = torch.full((32,), 7, dtype=torch.uint8, device="cuda")
        dst_ptr = 0xDEAD_BEEF

        def fake_read(session, read_ptrs, remote_ptrs, lengths):
            del session, remote_ptrs
            for read_ptr, length in zip(read_ptrs, lengths):
                MooncakeTransferVerifier._copy_device_to_device(
                    read_ptr,
                    src.data_ptr() + 32 - length,
                    length,
                )
            return 0

        engine.batch_transfer_sync_read.side_effect = fake_read

        verifier = MooncakeTransferVerifier(engine, torch.device("cuda"))
        assert verifier.verify_remote_visibility(
            "remote:1",
            [src.data_ptr()],
            [dst_ptr],
            [32],
        )

    def test_verify_fails_when_readback_never_matches(self, monkeypatch):
        monkeypatch.setenv("VLLM_MOONCAKE_TRANSFER_VERIFY_TAIL_BYTES", "8")
        monkeypatch.setenv("VLLM_MOONCAKE_TRANSFER_VERIFY_MAX_RETRIES", "2")
        monkeypatch.setenv("VLLM_MOONCAKE_TRANSFER_VERIFY_RETRY_SLEEP_S", "0")

        engine = MagicMock()
        engine.batch_register_memory.return_value = 0

        def fake_read(session, read_ptrs, remote_ptrs, lengths):
            del session, remote_ptrs, lengths
            for read_ptr in read_ptrs:
                dst = torch.zeros(8, dtype=torch.uint8, device="cuda")
                MooncakeTransferVerifier._copy_device_to_device(
                    read_ptr, dst.data_ptr(), 8
                )
            return 0

        engine.batch_transfer_sync_read.side_effect = fake_read

        verifier = MooncakeTransferVerifier(engine, torch.device("cuda"))
        src = torch.full((16,), 9, dtype=torch.uint8, device="cuda")

        assert not verifier.verify_remote_visibility(
            "remote:1",
            [src.data_ptr()],
            [0x1000],
            [16],
        )
