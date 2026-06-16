# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer_guard import (
    _VERIFY_EDGE_BYTES,
    MooncakeTransferVerifier,
    get_remote_session_lock,
)


class TestMooncakeSessionLock:
    def test_same_session_returns_same_lock(self):
        assert get_remote_session_lock("host:1234") is get_remote_session_lock(
            "host:1234"
        )

    def test_different_sessions_return_different_locks(self):
        assert get_remote_session_lock("host:1234") is not get_remote_session_lock(
            "host:5678"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMooncakeTransferVerifier:
    def _make_verifier(self):
        engine = MagicMock()
        engine.batch_register_memory.return_value = 0
        return engine, MooncakeTransferVerifier(engine, torch.device("cuda"))

    def test_verify_succeeds_when_remote_matches(self):
        engine, verifier = self._make_verifier()
        # length > 2 * edge -> head and tail are both checked.
        length = 4 * _VERIFY_EDGE_BYTES
        src = torch.arange(length, dtype=torch.uint8, device="cuda")
        dst_ptr = src.data_ptr()

        # Remote pointer arithmetic maps each edge back to its source offset.
        def fake_read(session, read_ptrs, remote_ptrs, lengths):
            del session
            for read_ptr, remote_ptr, seg_len in zip(read_ptrs, remote_ptrs, lengths):
                src_off = remote_ptr - dst_ptr
                MooncakeTransferVerifier._copy_device_to_device(
                    read_ptr, src.data_ptr() + src_off, seg_len
                )
            return 0

        engine.batch_transfer_sync_read.side_effect = fake_read

        assert verifier.verify_remote_visibility(
            "remote:1", [src.data_ptr()], [dst_ptr], [length]
        )

    def test_verify_fails_when_remote_is_zero(self, monkeypatch):
        import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer_guard as g

        monkeypatch.setattr(g, "_VERIFY_MAX_RETRIES", 2)
        monkeypatch.setattr(g, "_VERIFY_RETRY_SLEEP_S", 0.0)

        engine, verifier = self._make_verifier()
        src = torch.full((2 * _VERIFY_EDGE_BYTES,), 9, dtype=torch.uint8, device="cuda")

        def fake_read(session, read_ptrs, remote_ptrs, lengths):
            del session, remote_ptrs
            for read_ptr, seg_len in zip(read_ptrs, lengths):
                zeros = torch.zeros(seg_len, dtype=torch.uint8, device="cuda")
                MooncakeTransferVerifier._copy_device_to_device(
                    read_ptr, zeros.data_ptr(), seg_len
                )
            return 0

        engine.batch_transfer_sync_read.side_effect = fake_read

        assert not verifier.verify_remote_visibility(
            "remote:1", [src.data_ptr()], [0x1000], [2 * _VERIFY_EDGE_BYTES]
        )
