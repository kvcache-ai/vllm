# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDMA transfer visibility guards for MooncakeConnector PD transfers.

RDMA one-sided writes can return to the sender before the payload is visible
in the remote GPU's HBM, and concurrent writes to the same remote session may
race. The producer serializes writes per remote session and reads back both
ends of every descriptor until the remote bytes match the local source.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine

logger = init_logger(__name__)

# Bytes checked at both the head and tail of each descriptor. Tail catches
# truncated writes (NIC-to-HBM gap); head catches front-overwrite races.
_VERIFY_EDGE_BYTES = 64
# Read-back scratch buffer size (bytes) for one batch_transfer_sync_read call.
_VERIFY_SCRATCH_BYTES = 64 * 1024
_VERIFY_MAX_RETRIES = 500
_VERIFY_RETRY_SLEEP_S = 0.001

_session_locks: dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


def get_remote_session_lock(remote_session: str) -> threading.Lock:
    """Return a process-wide lock for transfers to one remote Mooncake session."""
    with _session_locks_guard:
        lock = _session_locks.get(remote_session)
        if lock is None:
            lock = threading.Lock()
            _session_locks[remote_session] = lock
        return lock


class MooncakeTransferVerifier:
    """Verify remote GPU visibility via RDMA read-back of descriptor edges."""

    def __init__(self, engine: TransferEngine, device: torch.device) -> None:
        self.engine = engine
        self.device = device
        # Serializes access to the shared scratch buffers across sender threads.
        self._lock = threading.Lock()
        self._scratch_bytes = _VERIFY_SCRATCH_BYTES
        self._read_buf = torch.empty(
            self._scratch_bytes, dtype=torch.uint8, device=device
        )
        self._expected_buf = torch.empty(
            self._scratch_bytes, dtype=torch.uint8, device=device
        )
        ret = self.engine.batch_register_memory(
            [self._read_buf.data_ptr()], [self._read_buf.nbytes]
        )
        if ret != 0:
            raise RuntimeError("Mooncake verifier scratch buffer registration failed.")

    def verify_remote_visibility(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> bool:
        """Poll until remote descriptor edges match local source, else fail."""
        edge = _VERIFY_EDGE_BYTES
        # (local_src_ptr, remote_dst_ptr, seg_len) for each edge to compare.
        segments: list[tuple[int, int, int]] = []
        for src, dst, length in zip(src_ptrs, dst_ptrs, lengths):
            if length <= 2 * edge:
                segments.append((src, dst, length))
            else:
                segments.append((src, dst, edge))
                segments.append((src + length - edge, dst + length - edge, edge))
        if not segments:
            return True

        with self._lock:
            for batch in _batch_segments_by_bytes(segments, self._scratch_bytes):
                if not self._verify_batch(remote_session, batch):
                    logger.error(
                        "Mooncake transfer remote visibility not confirmed after "
                        "%d retries (%d descriptors, session=%s)",
                        _VERIFY_MAX_RETRIES,
                        len(src_ptrs),
                        remote_session,
                    )
                    return False
        return True

    def _verify_batch(
        self, remote_session: str, segments: list[tuple[int, int, int]]
    ) -> bool:
        check_lens = [seg_len for _, _, seg_len in segments]
        remote_ptrs = [dst for _, dst, _ in segments]
        total = sum(check_lens)
        assert total <= self._scratch_bytes, (
            f"verify batch size {total} exceeds scratch {self._scratch_bytes}"
        )
        read_buf = self._read_buf[:total]
        expected_buf = self._expected_buf[:total]

        read_ptrs: list[int] = []
        offset = 0
        for src, _, seg_len in segments:
            self._copy_device_to_device(
                expected_buf.data_ptr() + offset, src, seg_len
            )
            read_ptrs.append(read_buf.data_ptr() + offset)
            offset += seg_len

        for attempt in range(_VERIFY_MAX_RETRIES):
            ret = self.engine.batch_transfer_sync_read(
                remote_session, read_ptrs, remote_ptrs, check_lens
            )
            if ret == 0:
                torch.cuda.synchronize(device=self.device)
                if torch.equal(read_buf, expected_buf):
                    if attempt > 0:
                        logger.debug(
                            "Mooncake transfer visibility confirmed after %d retries",
                            attempt + 1,
                        )
                    return True
            time.sleep(_VERIFY_RETRY_SLEEP_S)
        return False

    @staticmethod
    def _copy_device_to_device(dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        err = cudart.cudaMemcpy(
            dst_ptr,
            src_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        )[0]
        if err != 0:
            raise RuntimeError(f"cudaMemcpy D2D failed with error code {err}")


def _batch_segments_by_bytes(
    segments: list[tuple[int, int, int]], max_bytes: int
) -> list[list[tuple[int, int, int]]]:
    """Split segments so each batch fits in the scratch buffer."""
    batches: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    current_bytes = 0
    for src, dst, seg_len in segments:
        if seg_len > max_bytes:
            if current:
                batches.append(current)
                current = []
                current_bytes = 0
            batches.append([(src, dst, seg_len)])
            continue
        if current and current_bytes + seg_len > max_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append((src, dst, seg_len))
        current_bytes += seg_len
    if current:
        batches.append(current)
    return batches


def sync_device_after_remote_kv_write(device: torch.device) -> None:
    """Flush local GPU work before the decoder consumes received KV."""
    torch.cuda.synchronize(device=device)
