# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SGLang-style rid-based arrival doorbell for MooncakeConnector.

Background
----------
SGLang's ``MooncakeKVManager`` proves a transfer landed at the decoder
by tagging each request with a ``bootstrap_room`` (the rid) and routing
the per-request status through a separate, ordered channel
(``sync_status_to_decode_endpoint`` over ZMQ, plus the ``AUX_DATA``
RDMA push that carries the same room). The decoder treats the rid as
identity: only when the rid arrives back does the consumer mark the
request as truly received.

vLLM's MooncakeConnector already routes per-request status via
``MooncakeXferResponse.ok_reqs`` / ``err_reqs`` (req_id == rid). That
channel tells D *which* request the producer believes succeeded, but
it does not by itself prove the RDMA bytes for that request landed in
the decoder's HBM -- ``batch_transfer_sync_write`` returning zero only
means the local NIC has been informed, not that the remote memory was
updated. The original Qwen3-Omni KV cache corruption surfaced exactly
this gap (engine returned 0, decoder saw head/tail zeros).

Design
------
This module implements a small "arrival doorbell" mechanism aligned to
SGLang's rid pattern:

* D allocates an 8-byte slot in a GPU buffer registered with Mooncake
  and generates a random 64-bit nonce per request. The slot is zeroed.
* D ships ``(slot_addr, expected_nonce)`` to P inside
  ``MooncakeXferMetadata``.
* P stages the nonces in its own GPU scratch buffer (``NoncePad``) and
  appends one tiny descriptor per request to the end of its
  ``batch_transfer_sync_write`` call. RDMA WRITE ordering on a session
  guarantees the doorbell write retires after the preceding KV writes
  on the same destination memory region.
* After ZMQ ``ok_reqs`` arrives, D reads its own local slot and checks
  ``actual_nonce == expected_nonce``. A match proves the rid arrived,
  and -- by RDMA ordering -- that the preceding KV bytes did too. A
  mismatch demotes the request to ``finished_recving`` failure so the
  scheduler can retry rather than feed corrupted KV cache into decode.

The slot is intentionally tiny (16 bytes including 8-byte padding) so
the per-request overhead is one descriptor and 16 bytes of RDMA write.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine

logger = init_logger(__name__)

# Size of one doorbell slot in bytes. The first 8 bytes hold the
# little-endian nonce, the remaining 8 are reserved (zero) padding so
# slots stay aligned and we have room to grow the protocol later
# without breaking layout compatibility.
ARRIVAL_SLOT_BYTES = 16


@dataclass(frozen=True)
class DoorbellHandle:
    """Per-request handle returned by ``ArrivalDoorbell.allocate``.

    ``slot_addr`` is shipped to the producer; ``expected_nonce`` is the
    value the consumer expects to read back from the slot after the
    producer's RDMA write completes.
    """

    req_id: str
    slot_addr: int
    expected_nonce: int


class ArrivalDoorbell:
    """Consumer-side pool of arrival-doorbell slots on GPU memory.

    The whole buffer is registered with the Mooncake transfer engine
    once at construction so that producers can RDMA-write directly
    into individual slots.
    """

    def __init__(
        self,
        engine: TransferEngine,
        device: torch.device,
        capacity: int = 4096,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._device = device
        total_bytes = capacity * ARRIVAL_SLOT_BYTES
        self._buffer = torch.zeros(total_bytes, dtype=torch.uint8, device=device)
        ret = engine.batch_register_memory(
            [self._buffer.data_ptr()], [self._buffer.numel()]
        )
        if ret != 0:
            raise RuntimeError(
                f"Failed to register arrival-doorbell buffer with Mooncake (ret={ret})"
            )
        self._lock = threading.Lock()
        self._free: list[int] = list(range(capacity))
        # Maps req_id -> (slot_idx, expected_nonce).
        self._in_use: dict[str, tuple[int, int]] = {}

    def _slot_addr(self, slot_idx: int) -> int:
        return self._buffer.data_ptr() + slot_idx * ARRIVAL_SLOT_BYTES

    def allocate(self, req_id: str) -> DoorbellHandle:
        """Reserve a slot for ``req_id`` and return its handle."""
        with self._lock:
            if not self._free:
                raise RuntimeError(
                    "ArrivalDoorbell pool exhausted "
                    f"(capacity={self._capacity}, in_use={len(self._in_use)}). "
                    "Increase capacity or drain pending transfers."
                )
            if req_id in self._in_use:
                raise RuntimeError(
                    f"req_id {req_id} already has an allocated doorbell slot"
                )
            slot_idx = self._free.pop()
            # 8-byte nonce; the high bit is masked to keep the value
            # inside Python's signed-int safe range when round-tripping
            # through some msgspec encoders later.
            nonce = secrets.token_bytes(8)
            nonce_int = int.from_bytes(nonce, "little") & ((1 << 63) - 1)
            self._in_use[req_id] = (slot_idx, nonce_int)
        # Zero the slot on device; this happens outside the lock since
        # the slice is exclusive to this owner.
        start = slot_idx * ARRIVAL_SLOT_BYTES
        self._buffer[start : start + ARRIVAL_SLOT_BYTES].zero_()
        return DoorbellHandle(
            req_id=req_id,
            slot_addr=self._slot_addr(slot_idx),
            expected_nonce=nonce_int,
        )

    def has_slot(self, req_id: str) -> bool:
        """Return ``True`` iff ``req_id`` currently owns a slot."""
        with self._lock:
            return req_id in self._in_use

    def verify(self, req_id: str) -> bool:
        """Return ``True`` iff the consumer-side slot holds ``expected_nonce``.

        Returns ``False`` if the slot is unknown or the nonce does not
        match. This reads the device slot via a tiny D2H copy. Callers
        must only invoke verify *after* the producer has signalled
        success through the ZMQ status channel; otherwise the slot is
        racy.
        """
        with self._lock:
            entry = self._in_use.get(req_id)
        if entry is None:
            return False
        slot_idx, expected = entry
        start = slot_idx * ARRIVAL_SLOT_BYTES
        slot_bytes = bytes(self._buffer[start : start + 8].cpu().tolist())
        actual = int.from_bytes(slot_bytes, "little")
        if actual != expected:
            logger.warning(
                "Arrival doorbell mismatch for req=%s: expected=%x actual=%x",
                req_id,
                expected,
                actual,
            )
            return False
        return True

    def release(self, req_id: str) -> None:
        """Return the slot to the free pool. Safe to call twice."""
        with self._lock:
            entry = self._in_use.pop(req_id, None)
            if entry is not None:
                slot_idx, _ = entry
                self._free.append(slot_idx)

    @property
    def capacity(self) -> int:
        return self._capacity


class NoncePad:
    """Producer-side device-resident scratch buffer for staging nonces.

    Each call to ``stage`` reserves the next N slots (wrapping around
    the capacity), writes the supplied nonces to those slots, and
    returns the source pointers so the caller can append RDMA write
    descriptors targeting the matching ``ArrivalDoorbell`` slots on
    the consumer.

    Slot reuse is by ring buffer: the pad assumes the previous use of
    each slot has fully retired by the time the producer wraps. With a
    capacity an order of magnitude larger than the producer's
    concurrent transfer fan-out this is safe in practice (the pad is
    on the *send* side of an already-completed batch by the time we
    reuse the slot).
    """

    def __init__(
        self,
        engine: TransferEngine,
        device: torch.device,
        capacity: int = 4096,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        total_bytes = capacity * ARRIVAL_SLOT_BYTES
        self._buffer = torch.zeros(total_bytes, dtype=torch.uint8, device=device)
        ret = engine.batch_register_memory(
            [self._buffer.data_ptr()], [self._buffer.numel()]
        )
        if ret != 0:
            raise RuntimeError(
                f"Failed to register nonce pad with Mooncake (ret={ret})"
            )
        self._lock = threading.Lock()
        self._cursor = 0

    def _slot_addr(self, slot_idx: int) -> int:
        return self._buffer.data_ptr() + slot_idx * ARRIVAL_SLOT_BYTES

    def stage(self, nonces: list[int]) -> list[int]:
        """Write ``nonces`` to consecutive slots and return src pointers."""
        if not nonces:
            return []
        n = len(nonces)
        with self._lock:
            slots = [(self._cursor + i) % self._capacity for i in range(n)]
            self._cursor = (self._cursor + n) % self._capacity
        # Build a tiny host tensor and one D2D-style copy per slot. The
        # slots may wrap so we just iterate; the bytes-per-stage is
        # ``ARRIVAL_SLOT_BYTES * n`` which is negligible compared to KV
        # transfer sizes.
        for slot_idx, nonce in zip(slots, nonces):
            payload = nonce.to_bytes(8, "little") + b"\x00" * (ARRIVAL_SLOT_BYTES - 8)
            staged = torch.tensor(
                list(payload), dtype=torch.uint8, device=self._buffer.device
            )
            start = slot_idx * ARRIVAL_SLOT_BYTES
            self._buffer[start : start + ARRIVAL_SLOT_BYTES] = staged
        return [self._slot_addr(s) for s in slots]

    @property
    def capacity(self) -> int:
        return self._capacity
