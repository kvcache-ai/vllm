# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RDMA transfer visibility guards for MooncakeConnector PD transfers.

RDMA one-sided writes can return to the sender before the payload is visible
in the remote GPU's HBM. Under concurrent batch transfers to the same remote
session, completion ordering is also not guaranteed unless transfers are
serialized. These helpers address both issues.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.logger import init_logger

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine

logger = init_logger(__name__)

# Bytes read back from the tail of each descriptor when verifying visibility.
_DEFAULT_VERIFY_TAIL_BYTES = 64
# Upper bound on descriptors per batch_transfer_sync_write call.
_MAX_DESCRIPTORS_PER_BATCH = 512

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


def mooncake_transfer_verify_enabled() -> bool:
    return envs.VLLM_MOONCAKE_TRANSFER_VERIFY


def _verify_tail_bytes() -> int:
    return envs.VLLM_MOONCAKE_TRANSFER_VERIFY_TAIL_BYTES


def _verify_max_retries() -> int:
    return envs.VLLM_MOONCAKE_TRANSFER_VERIFY_MAX_RETRIES


def _verify_retry_sleep_s() -> float:
    return envs.VLLM_MOONCAKE_TRANSFER_VERIFY_RETRY_SLEEP_S


class MooncakeTransferVerifier:
    """Verify remote GPU visibility via RDMA read-back of descriptor tails."""

    def __init__(self, engine: TransferEngine, device: torch.device) -> None:
        self.engine = engine
        self.device = device
        tail_bytes = _verify_tail_bytes()
        scratch_bytes = 2 * _MAX_DESCRIPTORS_PER_BATCH * tail_bytes
        self._scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)
        ret = self.engine.batch_register_memory(
            [self._scratch.data_ptr()], [self._scratch.nbytes]
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
        """Poll until remote tails match local source tails, or retries exhaust."""
        if not src_ptrs:
            return True

        tail_bytes = _verify_tail_bytes()
        check_lens: list[int] = []
        local_src_tails: list[int] = []
        remote_dst_tails: list[int] = []
        for src, dst, length in zip(src_ptrs, dst_ptrs, lengths):
            check_len = min(length, tail_bytes)
            check_lens.append(check_len)
            local_src_tails.append(src + length - check_len)
            remote_dst_tails.append(dst + length - check_len)

        max_retries = _verify_max_retries()
        retry_sleep = _verify_retry_sleep_s()
        total_bytes = sum(check_lens)
        read_buf = self._scratch[:total_bytes]
        expected_buf = self._scratch[total_bytes : 2 * total_bytes]

        read_ptrs: list[int] = []
        read_offset = 0
        for check_len in check_lens:
            read_ptrs.append(read_buf.data_ptr() + read_offset)
            read_offset += check_len

        expected_offset = 0
        for src_tail, check_len in zip(local_src_tails, check_lens):
            self._copy_device_to_device(
                dst_ptr=expected_buf.data_ptr() + expected_offset,
                src_ptr=src_tail,
                nbytes=check_len,
            )
            expected_offset += check_len

        for attempt in range(max_retries):
            ret = self.engine.batch_transfer_sync_read(
                remote_session, read_ptrs, remote_dst_tails, check_lens
            )
            if ret != 0:
                logger.warning(
                    "Mooncake transfer verify read-back failed (ret=%s, attempt=%d)",
                    ret,
                    attempt + 1,
                )
            elif torch.equal(read_buf, expected_buf):
                if attempt > 0:
                    logger.debug(
                        "Mooncake transfer remote visibility confirmed after %d retries",
                        attempt + 1,
                    )
                return True

            time.sleep(retry_sleep)

        logger.error(
            "Mooncake transfer remote visibility not confirmed after %d retries "
            "(%d descriptors, session=%s)",
            max_retries,
            len(src_ptrs),
            remote_session,
        )
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


def sync_device_after_remote_kv_write(device: torch.device) -> None:
    """Ensure the local GPU observes completed remote RDMA writes."""
    torch.cuda.synchronize(device=device)
