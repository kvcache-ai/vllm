# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Encoder-cache (EC) connector backed by Mooncake TransferEngine.

Used in disaggregated setups where an encoder / prefill instance produces
multimodal encoder outputs and a decode instance loads them over RDMA-capable
Mooncake transport instead of shared filesystem.
"""

from __future__ import annotations

import bisect
import math
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.mooncake._availability import (
    ensure_mooncake_available,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import MooncakeECConfig
from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ConsumerControlServer,
    ControlClient,
    ControlCompletion,
    ShardTopology,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_T = TypeVar("_T")

_LEASE_TTL_SECONDS = 300
_RESERVATION_REFRESH_SECONDS = _LEASE_TTL_SECONDS / 2
# A cancelled transfer stays on the scheduler's ignore list for as long as the
# worker refuses to reserve it again. The count is a backstop for a rate that
# outruns that TTL; the race it guards is a single drain interval wide.
_MAX_CANCELLED_TRANSFER_IDS = 1 << 16


@dataclass
class _ConsumerPoolAllocation:
    offset: int
    size: int
    tensor: torch.Tensor


@dataclass
class _PushReservation:
    mm_hash: str
    reservation_id: str
    allocation: _ConsumerPoolAllocation
    shape: tuple[int, ...]
    dtype: str
    ready: bool = False
    owns_allocation: bool = True
    discard_on_complete: bool = False
    created_at: float = field(default_factory=time.monotonic)
    expires_at: float = 0


@dataclass
class _PendingPush:
    tensor: torch.Tensor
    spec: ECMooncakePushSpec
    reservation: Future[list[dict[str, Any]]]
    ready_event: torch.Event | None
    enqueued_at: float


@dataclass
class _PushPerfWindow:
    started_at: float = field(default_factory=time.monotonic)
    batches: int = 0
    items: int = 0
    bytes: int = 0
    skipped_items: int = 0
    failures: int = 0
    stage_totals_ms: dict[str, float] = field(default_factory=dict)
    stage_max_ms: dict[str, float] = field(default_factory=dict)


class _ResidentPool(Generic[_T]):
    """Content-addressed entries kept until their space is needed.

    Both sides of the connector hold the same thing under different names: a
    map from mm_hash to a device resource, a count of who is using it, and an
    eviction order over the rest. This is `BlockPool`'s accounting for
    variable-sized entries: `acquire`/`release` mirror `touch`/`free_blocks`,
    and `evict_lru` mirrors the reclaim inside `get_new_blocks`.

    An unreferenced entry stays resident. Eviction is driven by pressure, so
    the entry serves whoever needs it next instead of being transferred again.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0
        self._entries: dict[str, tuple[_T, int]] = {}
        self._refs: Counter[str] = Counter()
        # Unreferenced entries in eviction order, oldest first.
        self._evictable: OrderedDict[str, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    @property
    def num_evictable(self) -> int:
        return len(self._evictable)

    def referenced(self) -> list[str]:
        """Keys that are in use. `_refs` only holds entries above zero."""
        return list(self._refs)

    def referenced_or_retired(self) -> list[str]:
        """Every key held, in insertion order."""
        return list(self._entries)

    def get(self, key: str) -> _T | None:
        entry = self._entries.get(key)
        return entry[0] if entry is not None else None

    def insert(self, key: str, value: _T, nbytes: int) -> None:
        """Add a referenced entry, replacing any previous one."""
        previous = self._entries.get(key)
        if previous is not None:
            self.used -= previous[1]
        self._entries[key] = (value, nbytes)
        self.used += nbytes
        self.pin(key)

    def pin(self, key: str) -> _T | None:
        """Mark an entry as in use without counting a new reference.

        For a holder whose references are discovered by scanning rather than
        released in pairs, `pin`/`retire` are the matching operations.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._evictable.pop(key, None)
        self._refs[key] = max(1, self._refs[key])
        return entry[0]

    def retire(self, key: str) -> None:
        """Drop every reference; the entry is evictable from now on."""
        if key not in self._entries:
            return
        self._refs.pop(key, None)
        self._evictable[key] = None

    def refresh(self, key: str) -> None:
        """Move an unreferenced entry to the back of the eviction order."""
        if key in self._evictable:
            self._evictable.move_to_end(key)

    def acquire(self, key: str) -> _T | None:
        """Take one reference so pressure cannot evict the entry."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._evictable.pop(key, None)
        self._refs[key] += 1
        return entry[0]

    def release(self, key: str) -> None:
        """Drop one reference; the entry becomes evictable at zero."""
        if key not in self._entries:
            return
        count = self._refs[key] - 1
        if count > 0:
            self._refs[key] = count
            return
        self._refs.pop(key, None)
        self._evictable[key] = None

    def evict_lru(self, evict: Callable[[str, _T], bool]) -> str | None:
        """Drop the oldest entry `evict` accepts, and return its key.

        `evict` returns False for an entry that cannot go yet (a lease the
        remote side still holds, a deregistration that failed). Those keep
        their place in the order and the next candidate is tried.
        """
        for key in list(self._evictable):
            value, nbytes = self._entries[key]
            if not evict(key, value):
                continue
            self._evictable.pop(key, None)
            del self._entries[key]
            self._refs.pop(key, None)
            self.used -= nbytes
            return key
        return None

    def clear(self) -> None:
        self._entries.clear()
        self._refs.clear()
        self._evictable.clear()
        self.used = 0


class _ContiguousAllocator:
    def __init__(self, capacity: int, alignment: int = 256):
        self.capacity = capacity
        self.alignment = alignment
        self._free = [(0, capacity)]

    def allocate(self, nbytes: int) -> tuple[int, int] | None:
        size = math.ceil(nbytes / self.alignment) * self.alignment
        for index, (offset, available) in enumerate(self._free):
            if size > available:
                continue
            if size == available:
                self._free.pop(index)
            else:
                self._free[index] = (offset + size, available - size)
            return offset, size
        return None

    def free(self, offset: int, size: int) -> None:
        index = bisect.bisect_left(self._free, (offset, size))
        self._free.insert(index, (offset, size))
        # Coalesce with the neighbours only; the rest of the list is already
        # merged, so a full re-scan per free is wasted work.
        if index + 1 < len(self._free):
            next_offset, next_size = self._free[index + 1]
            if offset + size == next_offset:
                self._free[index] = (offset, size + next_size)
                self._free.pop(index + 1)
        if index > 0:
            previous_offset, previous_size = self._free[index - 1]
            current_offset, current_size = self._free[index]
            if previous_offset + previous_size == current_offset:
                self._free[index - 1] = (
                    previous_offset,
                    previous_size + current_size,
                )
                self._free.pop(index)


class ECMooncakeWorker:
    """
    EC connector using Mooncake TransferEngine for GPU tensor transport.

    The producer pushes each encoder output into a receive buffer the consumer
    reserved for it, so the transfer overlaps encoding instead of waiting for
    the consumer to ask. An item the consumer's encoder cache evicted stays in
    that pool and is handed back locally; when neither has it, the load fails
    with a retryable error so the caller can re-issue the request.

    Extra config (``ec_connector_extra_config``):

    - ``mooncake_protocol`` (optional): Passed to ``TransferEngine.initialize``
      (default ``"rdma"``).
    - ``consumer_buffer_pool_size`` (consumer, optional): Bytes reserved for a
      long-lived registered CUDA receive arena (default ``ec_buffer_size``).
    - ``reservation_zmq_port`` (consumer worker, required): Exposes registered
      receive addresses over ZMQ. Replica ``d`` of the first pipeline stage owns
      the block starting at ``port + d * tensor_parallel_size``; tensor-parallel
      rank ``r`` in that block listens on ``block + r``, and rank 0 reports the
      whole block, so a producer only needs the block's first address.
    - ``reservation_zmq_addr`` (consumer scheduler, required): Address of the
      consumer control channel. Defaults to ``tcp://127.0.0.1:<port>``.
    - ``transfer_max_workers`` (optional): Maximum concurrent Mooncake transfer
      batches (default ``4``).
    - ``control_max_workers`` (optional): Maximum concurrent reservation requests
      issued by a producer (default ``8``).
    - ``transfer_metrics_log_interval`` (optional): Seconds between aggregated
      push-transfer performance logs (default ``10``; ``0`` disables them).
    - ``consumer_metrics_log_interval`` (optional): Seconds between aggregated
      consumer lifecycle logs (default ``10``; ``0`` disables them).

    Parallelism: consumers may use tensor, pipeline and data parallelism.
    Producers must be unsharded and unreplicated: one copy of each encoder
    output is held and addressed directly, so splitting the producer would only
    duplicate the push.

    Only the first pipeline stage holds encoder outputs, and each tensor-parallel
    rank there gathers from its own cache, so every rank exposes a control
    channel and the producer writes into all of them concurrently from one
    registered source. That costs bandwidth but not latency, and avoids the
    second hop a receive-then-broadcast would add.

    Data parallelism additionally requires the caller to route both halves of a
    request to the same replica, because a push has to land where the request
    will run. The proxy is the only component that knows which replica it picked:
    it names the replica to the consumer (``X-data-parallel-rank``) and passes
    that replica's control address to the producer. Getting this wrong is loud
    rather than silent -- the replica that runs the request never sees its
    embedding and gives up after ``push_wait_timeout_s`` -- but it is the
    caller's responsibility, not something this connector can detect.
    """

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> ECMooncakeWorker:
        ensure_mooncake_available()
        config = MooncakeECConfig.from_vllm_config(vllm_config, ECConnectorRole.WORKER)
        hostname = get_ip()
        control_client = ControlClient(config.control_timeout_ms)
        try:
            return cls(
                config,
                hostname,
                control_client,
                ShardTopology(control_client),
            )
        except Exception:
            control_client.close()
            raise

    def __init__(
        self,
        config: MooncakeECConfig,
        hostname: str,
        control_client: ControlClient,
        topology: ShardTopology,
    ) -> None:
        self.is_producer = config.is_producer
        self.is_consumer = config.is_consumer
        self._buffer_device = config.buffer_device
        self._reservation_zmq_port = config.reservation_port
        self._consumer_pool_capacity = config.consumer_pool_size
        self._consumer_pool: torch.Tensor | None = None
        self._consumer_pool_allocator: _ContiguousAllocator | None = None
        # The receive pool is orders of magnitude larger than the encoder
        # cache, so an item the encoder cache evicted stays resident here and
        # a later request gets it for a dict lookup instead of a transfer.
        self._consumer_residents: _ResidentPool[_ConsumerPoolAllocation] = (
            _ResidentPool(self._consumer_pool_capacity)
        )
        self._consumer_retire_events: dict[str, torch.Event] = {}
        self._consumer_pending_frees: list[
            tuple[torch.Event, _ConsumerPoolAllocation]
        ] = []
        self._consumer_reclaimed: set[str] = set()
        self._consumer_rank_resolved = False
        self._is_receiving_rank = True
        self._tp_rank = 0
        self._tp_size = 1
        self._consumer_pool_disabled = False
        self._consumer_lock = threading.Lock()
        self._push_reservations: dict[str, _PushReservation] = {}
        self._cancelled_transfers: OrderedDict[str, float] = OrderedDict()
        self._control_server: ConsumerControlServer | None = None
        self._consumer_metrics_log_interval = config.consumer_metrics_log_interval
        self._consumer_metrics_started_at = time.monotonic()
        self._consumer_worker_metrics: Counter[str] = Counter()
        self._active_push_sources: Counter[tuple[str, int]] = Counter()
        self._active_push_sources_lock = threading.Lock()

        # Worker producer
        self._transfer = MooncakeTransfer(hostname, config.protocol)
        self._producer_pool_capacity = config.producer_pool_size
        self._producer_pool: torch.Tensor | None = None
        self._producer_pool_allocator: _ContiguousAllocator | None = None
        self._producer_pool_disabled = False
        self._producer_pool_lock = threading.Lock()
        self._transfer_metrics_log_interval = config.transfer_metrics_log_interval
        self._control_client = control_client
        self._topology = topology
        self._producer_metrics: Counter[str] = Counter()
        self._io_executor = ThreadPoolExecutor(
            max_workers=config.transfer_workers,
            thread_name_prefix="ec-mooncake-transfer",
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=config.control_workers,
            thread_name_prefix="ec-mooncake-control",
        )
        self._shard_pool: ThreadPoolExecutor | None = None
        self._shard_pool_lock = threading.Lock()
        self._pending_saves: list[tuple[str, Future[None]]] = []
        self._pending_reservations: dict[
            str, deque[tuple[ECMooncakePushSpec, Future[list[dict[str, Any]]]]]
        ] = {}
        self._pending_pushes: list[_PendingPush] = []
        self._push_perf_lock = threading.Lock()
        self._push_perf = _PushPerfWindow()
        self._active_transfer_batches = 0
        self._queued_transfer_batches = 0
        self._completed_loads: set[str] = set()
        self._failed_loads: set[str] = set()
        self._shutdown = False

    def _resolve_consumer_rank(self) -> None:
        """Place this worker in the consumer's receive topology.

        Encoder outputs only exist on the first pipeline stage, and every
        tensor-parallel rank there gathers from its own cache, so each of them
        receives its own copy on its own control channel. Ports run
        consecutively from the configured one so a producer holding the first
        address can reach the rest.
        """
        if self._consumer_rank_resolved:
            return
        self._consumer_rank_resolved = True
        try:
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            tp_group = get_tp_group()
            self._tp_rank = tp_group.rank_in_group
            self._tp_size = tp_group.world_size
            self._is_receiving_rank = get_pp_group().is_first_rank
        except AssertionError:
            # Groups are only absent outside a distributed run, where this
            # worker is the whole consumer.
            self._tp_rank = 0
            self._tp_size = 1
            self._is_receiving_rank = True

    def start_services(self) -> None:
        if (
            not self.is_consumer
            or self._reservation_zmq_port is None
            or self._control_server is not None
        ):
            return
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Later pipeline stages hold no encoder outputs, so they need
            # neither a receive pool nor a control channel.
            return
        raw_device = self._buffer_device
        device_name = (
            raw_device.lower() if isinstance(raw_device, str) and raw_device else "cuda"
        )
        self._ensure_consumer_pool(torch.device(device_name), allow_host=True)
        if self._consumer_pool is None:
            raise RuntimeError(
                "Mooncake push mode requires a registered consumer buffer pool."
            )
        base_port = self._reservation_zmq_port
        self._control_server = ConsumerControlServer(
            "0.0.0.0",
            base_port + self._tp_rank,
            self._reserve_push_destination,
            self._push_status,
            self._complete_push,
            self._cancel_push,
            self._expire_push_reservations,
            self._consumer_metrics_log_interval,
            peer_ports=[base_port + rank for rank in range(self._tp_size)],
            device=self._consumer_pool.device,
        )
        try:
            self._control_server.start()
        except Exception:
            self._control_server.close()
            self._control_server = None
            raise

    def _ensure_consumer_pool(
        self, device: torch.device, *, allow_host: bool = False
    ) -> None:
        if (
            self._consumer_pool is not None
            or self._consumer_pool_disabled
            or (device.type != "cuda" and not allow_host)
        ):
            return
        try:
            pool = torch.empty(
                self._consumer_pool_capacity, dtype=torch.uint8, device=device
            )
            if self._is_receiving_rank:
                # Producers write into this pool directly, so it needs a memory
                # region. Later pipeline stages never receive and skip it.
                ret = self._transfer.register_memory(pool)
                if ret != 0:
                    raise RuntimeError(f"Mooncake returned {ret}")
        except (RuntimeError, torch.OutOfMemoryError) as e:
            self._consumer_pool_disabled = True
            logger.warning(
                "Could not initialize the EC consumer buffer pool; falling back "
                "to per-tensor registration: %s",
                e,
            )
            return
        self._consumer_pool = pool
        self._consumer_pool_allocator = _ContiguousAllocator(pool.nbytes)
        logger.info(
            "Prepared %d-byte CUDA receive pool for Mooncake EC (registered=%s)",
            pool.nbytes,
            self._is_receiving_rank,
        )

    def _ensure_producer_pool(self, device: torch.device) -> None:
        """Register one staging slab so pushes never register per transfer.

        Registering the encoder output itself costs more than the transfer
        (register+unregister dominated the push path); staging into a slab
        that is registered once trades that for a device-to-device copy.
        """
        if self._producer_pool is not None or self._producer_pool_disabled:
            return
        with self._producer_pool_lock:
            if self._producer_pool is not None or self._producer_pool_disabled:
                return
            try:
                pool = torch.empty(
                    self._producer_pool_capacity, dtype=torch.uint8, device=device
                )
                ret = self._transfer.register_memory(pool)
                if ret != 0:
                    raise RuntimeError(f"Mooncake returned {ret}")
            except (RuntimeError, torch.OutOfMemoryError) as e:
                self._producer_pool_disabled = True
                logger.warning(
                    "Could not initialize the EC producer staging pool; falling "
                    "back to per-transfer registration: %s",
                    e,
                )
                return
            self._producer_pool = pool
            self._producer_pool_allocator = _ContiguousAllocator(pool.nbytes)
            logger.info(
                "Registered %d-byte staging pool for Mooncake EC pushes",
                pool.nbytes,
            )

    def _stage_push_sources(
        self, tensors: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]] | None:
        """Copy the batch into the staging pool; None if it does not fit."""
        if not tensors:
            return [], []
        self._ensure_producer_pool(tensors[0].device)
        pool = self._producer_pool
        allocator = self._producer_pool_allocator
        if pool is None or allocator is None:
            return None
        staged: list[torch.Tensor] = []
        regions: list[tuple[int, int]] = []
        with self._producer_pool_lock:
            for tensor in tensors:
                region = allocator.allocate(tensor.nbytes)
                if region is None:
                    for offset, size in regions:
                        allocator.free(offset, size)
                    return None
                regions.append(region)
                offset = region[0]
                staged.append(
                    pool.narrow(0, offset, tensor.nbytes)
                    .view(tensor.dtype)
                    .view(tensor.shape)
                )
        for destination, source in zip(staged, tensors):
            destination.copy_(source, non_blocking=True)
        return staged, regions

    def _release_push_staging(self, regions: list[tuple[int, int]]) -> None:
        allocator = self._producer_pool_allocator
        if allocator is None or not regions:
            return
        with self._producer_pool_lock:
            for offset, size in regions:
                allocator.free(offset, size)

    def _poll_consumer_pool_frees(self) -> None:
        allocator = self._consumer_pool_allocator
        if allocator is None:
            return
        with self._consumer_lock:
            pending = []
            for event, allocation in self._consumer_pending_frees:
                if event.query():
                    allocator.free(allocation.offset, allocation.size)
                else:
                    pending.append((event, allocation))
            self._consumer_pending_frees = pending

    def _reclaim_residents_locked(
        self, allocator: _ContiguousAllocator, nbytes: int
    ) -> tuple[int, int] | None:
        """Give up retired items, oldest first, until `nbytes` fits.

        Called only when the pool cannot satisfy an allocation, so a retired
        item survives until its memory is genuinely needed.
        """

        def evict(mm_hash: str, allocation: _ConsumerPoolAllocation) -> bool:
            event = self._consumer_retire_events.pop(mm_hash, None)
            if event is None or event.query():
                allocator.free(allocation.offset, allocation.size)
            else:
                self._consumer_pending_frees.append((event, allocation))
            self._consumer_reclaimed.add(mm_hash)
            self._consumer_worker_metrics["residents_reclaimed"] += 1
            return True

        while self._consumer_residents.evict_lru(evict) is not None:
            region = allocator.allocate(nbytes)
            if region is not None:
                return region
        return None

    def _take_resident_tensor(self, spec: ECMooncakeLoadSpec) -> torch.Tensor | None:
        """Hand back a copy the pool still holds.

        Retired and in-use entries live in the same map, so an item a later
        push reserved again still serves this load.
        """
        with self._consumer_lock:
            allocation = self._consumer_residents.get(spec.mm_hash)
            if allocation is None:
                self._consumer_worker_metrics["residents_missed"] += 1
                return None
            tensor = allocation.tensor
            if (
                tuple(tensor.shape) != tuple(spec.shape)
                or str(tensor.dtype).split(".")[-1] != spec.dtype
            ):
                self._consumer_worker_metrics["residents_mismatched"] += 1
                return None
            self._consumer_residents.pin(spec.mm_hash)
            self._consumer_retire_events.pop(spec.mm_hash, None)
            self._consumer_worker_metrics["residents_promoted"] += 1
            return tensor

    def _release_stale_consumer_allocations(
        self, encoder_cache: dict[str, torch.Tensor]
    ) -> None:
        if self._consumer_pool is None:
            return
        with self._consumer_lock:
            reserved_allocations = {
                id(reservation.allocation)
                for reservation in self._push_reservations.values()
            }
            # Walk only the referenced entries: the retired set grows to
            # thousands and none of it can change state here.
            for mm_hash in self._consumer_residents.referenced():
                allocation = self._consumer_residents.get(mm_hash)
                if allocation is None:
                    continue
                if encoder_cache.get(mm_hash) is allocation.tensor:
                    continue
                if id(allocation) in reserved_allocations:
                    continue
                # Retire rather than free: the bytes stay valid and serve the
                # next request that needs this item. The event orders the
                # eventual reuse behind whatever still reads the tensor.
                event = torch.Event()
                event.record(
                    torch.accelerator.current_stream(self._consumer_pool.device)
                )
                self._consumer_retire_events[mm_hash] = event
                self._consumer_residents.retire(mm_hash)
                self._consumer_worker_metrics["residents_retired"] += 1
        self._poll_consumer_pool_frees()

    @staticmethod
    def _hash_samples(values: list[str], limit: int = 5) -> list[str]:
        return [value[:16] for value in values[:limit]]

    def _maybe_log_consumer_worker_metrics(self) -> None:
        now = time.monotonic()
        if (
            self._consumer_metrics_log_interval <= 0
            or now - self._consumer_metrics_started_at
            < self._consumer_metrics_log_interval
        ):
            return
        with self._consumer_lock:
            ready = [
                mm_hash
                for mm_hash, reservation in self._push_reservations.items()
                if reservation.ready
            ]
            pending = [
                mm_hash
                for mm_hash, reservation in self._push_reservations.items()
                if not reservation.ready
            ]
            metrics = dict(self._consumer_worker_metrics)
            self._consumer_worker_metrics.clear()
            residents = len(self._consumer_residents)
            live = len(self._consumer_residents.referenced())
            retired = self._consumer_residents.num_evictable
            pending_frees = len(self._consumer_pending_frees)
            oldest_reservation_ms = max(
                (
                    (now - reservation.created_at) * 1000
                    for reservation in self._push_reservations.values()
                ),
                default=0.0,
            )
        logger.info(
            "EC Mooncake consumer worker: lifecycle=%s, reservations_ready=%d, "
            "reservations_pending=%d, residents=%d, live=%d, retired=%d, "
            "pending_frees=%d, "
            "oldest_reservation_ms=%.1f, ready_hashes=%s, pending_hashes=%s",
            metrics,
            len(ready),
            len(pending),
            residents,
            live,
            retired,
            pending_frees,
            oldest_reservation_ms,
            self._hash_samples(ready),
            self._hash_samples(pending),
        )
        self._consumer_metrics_started_at = now

    @staticmethod
    def _expire_cancel_records(records: OrderedDict[str, float], now: float) -> int:
        """Drop the cancels that can no longer be told apart from unknown ids.

        Both roles keep one record per multimodal item they handle, and both
        consult it on a per-item hot path, so a full rescan costs the square
        of the item rate: at 53 items/s the worker's 300 s window is 16k
        entries and its sweep ran under `_consumer_lock` on every
        reservation. Callers append in deadline order -- `move_to_end` when
        refreshing one -- so the front is always the oldest and the sweep
        stops at the first live entry.

        Returns:
            How many records were dropped.
        """
        dropped = 0
        while records:
            expires_at = next(iter(records.values()))
            if expires_at > now and len(records) <= _MAX_CANCELLED_TRANSFER_IDS:
                break
            records.popitem(last=False)
            dropped += 1
        return dropped

    def _expire_push_reservations_locked(self) -> None:
        now = time.monotonic()
        allocator = self._consumer_pool_allocator
        assert allocator is not None
        for transfer_id, reservation in list(self._push_reservations.items()):
            if reservation.expires_at > now:
                continue
            if reservation.owns_allocation:
                allocator.free(
                    reservation.allocation.offset, reservation.allocation.size
                )
            self._push_reservations.pop(transfer_id)
            self._consumer_worker_metrics["reservations_expired"] += 1
        self._consumer_worker_metrics["cancel_records_dropped"] += (
            self._expire_cancel_records(self._cancelled_transfers, now)
        )

    def _expire_push_reservations(self) -> int:
        with self._consumer_lock:
            before = len(self._push_reservations)
            self._expire_push_reservations_locked()
            return before - len(self._push_reservations)

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        transfer_id = str(payload["transfer_id"])
        mm_hash = str(payload["mm_hash"])
        nbytes = int(payload["nbytes"])
        shape = tuple(int(value) for value in payload["shape"])
        dtype_name = str(payload["dtype"])
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype string: {dtype_name!r}")
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if expected_nbytes != nbytes:
            raise ValueError("shape and dtype do not match nbytes")

        with self._consumer_lock:
            self._expire_push_reservations_locked()
            if transfer_id in self._cancelled_transfers:
                self._consumer_worker_metrics["reservations_cancelled_early"] += 1
                return {
                    "reservation_id": "",
                    "dst_session": "",
                    "dst_ptr": 0,
                    "nbytes": nbytes,
                    "write": False,
                    "ready": False,
                    "cancelled": True,
                }
            existing = self._push_reservations.get(transfer_id)
            if existing is not None:
                if (
                    existing.mm_hash != mm_hash
                    or existing.shape != shape
                    or existing.dtype != dtype_name
                ):
                    raise ValueError("conflicting reservation for transfer_id")
                reservation = existing
                should_write = False
                key = (
                    "reservations_reused_ready"
                    if existing.ready
                    else ("reservations_reused_pending")
                )
                self._consumer_worker_metrics[key] += 1
                if not existing.ready:
                    existing.expires_at = time.monotonic() + _LEASE_TTL_SECONDS
            else:
                cached = self._consumer_residents.get(mm_hash)
                if cached is not None:
                    if (
                        tuple(cached.tensor.shape) != shape
                        or cached.tensor.dtype != dtype
                    ):
                        raise ValueError("conflicting cached tensor for mm_hash")
                    reservation = _PushReservation(
                        mm_hash=mm_hash,
                        reservation_id=uuid.uuid4().hex,
                        allocation=cached,
                        shape=shape,
                        dtype=dtype_name,
                        ready=True,
                        owns_allocation=False,
                        expires_at=time.monotonic() + _LEASE_TTL_SECONDS,
                    )
                    should_write = False
                    # Live again: it must not be reclaimed under pressure.
                    self._consumer_residents.pin(mm_hash)
                    self._consumer_retire_events.pop(mm_hash, None)
                    self._consumer_worker_metrics["reservations_cached"] += 1
                else:
                    pool = self._consumer_pool
                    allocator = self._consumer_pool_allocator
                    assert pool is not None and allocator is not None
                    region = allocator.allocate(nbytes)
                    if region is None:
                        self._expire_push_reservations_locked()
                        region = allocator.allocate(nbytes)
                    if region is None:
                        region = self._reclaim_residents_locked(allocator, nbytes)
                    if region is None:
                        raise RuntimeError("EC consumer buffer pool is full")
                    offset, size = region
                    tensor = pool.narrow(0, offset, nbytes).view(dtype).view(shape)
                    allocation = _ConsumerPoolAllocation(offset, size, tensor)
                    reservation = _PushReservation(
                        mm_hash=mm_hash,
                        reservation_id=uuid.uuid4().hex,
                        allocation=allocation,
                        shape=shape,
                        dtype=dtype_name,
                        expires_at=time.monotonic() + _LEASE_TTL_SECONDS,
                    )
                    should_write = True
                    self._consumer_worker_metrics["reservations_created"] += 1
                self._push_reservations[transfer_id] = reservation

        return {
            "reservation_id": reservation.reservation_id,
            "dst_session": self._transfer.local_session(),
            "dst_ptr": reservation.allocation.tensor.data_ptr(),
            "nbytes": reservation.allocation.tensor.nbytes,
            "write": should_write,
            "ready": reservation.ready,
            "cached": not reservation.owns_allocation,
        }

    def _push_status(self, transfer_id: str) -> dict[str, Any] | None:
        with self._consumer_lock:
            reservation = self._push_reservations.get(transfer_id)
            if reservation is None:
                return None
            return {
                "mm_hash": reservation.mm_hash,
                "ready": reservation.ready,
                "reservation_id": reservation.reservation_id,
                "nbytes": reservation.allocation.tensor.nbytes,
                "shape": list(reservation.shape),
                "dtype": reservation.dtype,
            }

    def _complete_push(
        self, transfer_id: str, reservation_id: str
    ) -> ControlCompletion:
        with self._consumer_lock:
            reservation = self._push_reservations.get(transfer_id)
            if reservation is None or reservation.reservation_id != reservation_id:
                self._consumer_worker_metrics["completions_rejected"] += 1
                return ControlCompletion(False)
            if reservation.ready:
                self._consumer_worker_metrics["completions_repeated"] += 1
                return ControlCompletion(True)
            self._consumer_worker_metrics["completions_accepted"] += 1
            if reservation.discard_on_complete:
                allocator = self._consumer_pool_allocator
                assert allocator is not None
                self._push_reservations.pop(transfer_id)
                if reservation.owns_allocation:
                    allocator.free(
                        reservation.allocation.offset, reservation.allocation.size
                    )
                self._consumer_worker_metrics["reservations_discarded"] += 1
                return ControlCompletion(True)
            reservation.ready = True
            reservation.expires_at = time.monotonic() + _LEASE_TTL_SECONDS
            return ControlCompletion(True, became_ready=True)

    def _cancel_push(
        self, transfer_id: str, reservation_id: str, abandon: bool = False
    ) -> bool:
        with self._consumer_lock:
            reservation = self._push_reservations.get(transfer_id)
            if (
                reservation is not None
                and reservation_id
                and reservation.reservation_id != reservation_id
            ):
                self._consumer_worker_metrics["cancellations_rejected"] += 1
                return False
            self._cancelled_transfers[transfer_id] = (
                time.monotonic() + _LEASE_TTL_SECONDS
            )
            self._cancelled_transfers.move_to_end(transfer_id)
            if reservation is None:
                self._consumer_worker_metrics["cancellations_pre_reserved"] += 1
                return True
            allocator = self._consumer_pool_allocator
            assert allocator is not None
            if not reservation.ready and not abandon:
                reservation.discard_on_complete = True
                self._consumer_worker_metrics["cancellations_deferred"] += 1
                return True
            self._push_reservations.pop(transfer_id)
            if reservation.owns_allocation:
                allocator.free(
                    reservation.allocation.offset, reservation.allocation.size
                )
            self._consumer_worker_metrics["reservations_cancelled"] += 1
            return True

    def _take_pushed_tensor(
        self, spec: ECMooncakeLoadSpec
    ) -> tuple[torch.Tensor, _ConsumerPoolAllocation]:
        with self._consumer_lock:
            reservation = self._push_reservations.get(spec.transfer_id)
            # Not compared against `spec.reservation_id`: each shard mints its
            # own, while the spec carries the one from whichever shard's event
            # the scheduler observed. `transfer_id` is assigned per request
            # item and is already unique, and a stale reservation for a reused
            # one is rejected by `_reserve_push_destination`.
            if reservation is None or not reservation.ready:
                self._consumer_worker_metrics["takes_rejected"] += 1
                raise RuntimeError(
                    f"Pushed EC tensor is not ready for mm_hash={spec.mm_hash}"
                )
            self._push_reservations.pop(spec.transfer_id)
            self._consumer_residents.insert(
                spec.mm_hash, reservation.allocation, reservation.allocation.size
            )
            self._consumer_worker_metrics["reservations_taken"] += 1
            return reservation.allocation.tensor, reservation.allocation

    def _shard_executor(self) -> ThreadPoolExecutor:
        """Threads for the extra shards of a sharded consumer.

        Reserving and writing both fan out from a task that already holds a
        worker of the control or transfer pool, so the extra shards need a
        pool of their own: queueing them behind their own caller deadlocks as
        soon as every worker there is waiting. Nothing submitted here fans out
        again, so this pool cannot deadlock on itself.
        """
        with self._shard_pool_lock:
            if self._shard_pool is None:
                self._shard_pool = ThreadPoolExecutor(
                    max_workers=32, thread_name_prefix="ec-mooncake-shard"
                )
            return self._shard_pool

    def _reserve_one(self, addr: str, spec: ECMooncakePushSpec) -> dict[str, Any]:
        result = self._control_client.request(
            addr,
            {
                "op": "reserve",
                "transfer_id": spec.transfer_id,
                "mm_hash": spec.mm_hash,
                "nbytes": spec.nbytes,
                "shape": list(spec.shape),
                "dtype": spec.dtype,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError("Invalid EC reservation response")
        result["_received_at"] = time.monotonic()
        result["addr"] = addr
        return result

    def _reserve_remote(self, spec: ECMooncakePushSpec) -> list[dict[str, Any]]:
        """Reserve a destination on every shard of the consumer."""
        shards = self._topology.shards(spec.consumer_zmq)
        if len(shards) == 1:
            return [self._reserve_one(shards[0], spec)]
        # This already runs on the control pool, so the extra shards go to the
        # fan-out pool: queueing them behind their own caller would deadlock
        # once every control worker is holding a reservation.
        extra = [
            self._shard_executor().submit(self._reserve_one, addr, spec)
            for addr in shards[1:]
        ]
        return [self._reserve_one(shards[0], spec)] + [f.result() for f in extra]

    def _cancel_remote(
        self, consumer_zmq: str, transfer_id: str, reservation_id: str
    ) -> bool:
        """Release this transfer on every shard that reserved for it.

        A sharded consumer holds one reservation per rank, so cancelling only
        the first would leave the rest pinning pool slots until they expire.
        """
        cancelled = False
        for addr in self._topology.shards(consumer_zmq):
            result = self._control_client.request(
                addr,
                {
                    "op": "cancel",
                    "transfer_id": transfer_id,
                    "reservation_id": reservation_id,
                },
            )
            cancelled |= isinstance(result, dict) and bool(result.get("cancelled"))
        return cancelled

    def start_save_caches(
        self,
        metadata: ECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        for spec in metadata.pushes:
            reservation = self._control_executor.submit(self._reserve_remote, spec)
            self._pending_reservations.setdefault(spec.mm_hash, deque()).append(
                (spec, reservation)
            )
        if not isinstance(encoder_cache, dict):
            return
        for mm_hash in dict.fromkeys(spec.mm_hash for spec in metadata.pushes):
            tensor = encoder_cache.get(mm_hash)
            if tensor is not None:
                self._submit_reserved_pushes(tensor, mm_hash)

    def start_load_caches(
        self,
        metadata: ECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Reached on steps with no work, from a stage that never gathers
            # multimodal embeddings. Taking a transfer here would fail for
            # want of a reservation and fail the load for everyone.
            return
        self._transfer.ensure_ready()
        raw_buf = self._buffer_device
        buf = raw_buf.lower() if isinstance(raw_buf, str) and raw_buf else "cuda"
        if buf == "cuda" and not torch.accelerator.is_available():
            raise RuntimeError(
                "ECMooncakeConnector requires CUDA for ec_buffer_device=cuda"
            )
        self._release_stale_consumer_allocations(encoder_cache)

        for spec in metadata.loads:
            if spec.mm_hash in encoder_cache:
                if spec.pushed:
                    # The spec's id is one shard's; cancel by transfer.
                    self._cancel_push(spec.transfer_id, "")
                self._completed_loads.add(spec.mm_hash)
                continue
            if spec.local:
                resident = self._take_resident_tensor(spec)
                if resident is None:
                    # Reclaimed before the scheduler heard about it; the load
                    # falls back to a transfer on a later step.
                    self._failed_loads.add(spec.mm_hash)
                else:
                    encoder_cache[spec.mm_hash] = resident
                    self._completed_loads.add(spec.mm_hash)
                continue
            if spec.pushed:
                try:
                    pushed_tensor, _ = self._take_pushed_tensor(spec)
                except RuntimeError as e:
                    logger.warning("EC Mooncake pushed load failed: %s", e)
                    self._failed_loads.add(spec.mm_hash)
                    continue
                encoder_cache[spec.mm_hash] = pushed_tensor
                self._completed_loads.add(spec.mm_hash)
                continue
            logger.warning(
                "EC Mooncake load for mm_hash=%s has no transfer to take",
                spec.mm_hash,
            )
            self._failed_loads.add(spec.mm_hash)

    def _push_batch(self, pushes: list[_PendingPush]) -> None:
        started_at = time.monotonic()
        with self._push_perf_lock:
            self._queued_transfer_batches -= 1
            self._active_transfer_batches += 1

        queue_waits_ms = [
            max(0, started_at - push.enqueued_at) * 1000 for push in pushes
        ]
        stage_ms = {
            "queue": sum(queue_waits_ms),
            "reserve": 0.0,
            "cuda": 0.0,
            "register": 0.0,
            "rdma": 0.0,
            "unregister": 0.0,
            "complete": 0.0,
        }
        ready: list[tuple[_PendingPush, dict[str, Any]]] = []
        notifications: list[tuple[_PendingPush, dict[str, Any]]] = []
        failed = False
        try:
            synchronized: set[int] = set()
            for push in pushes:
                stage_started_at = time.monotonic()
                reservations = push.reservation.result()
                stale = [
                    index
                    for index, shard in enumerate(reservations)
                    if not shard.get("ready", False)
                    and time.monotonic() - float(shard.get("_received_at", started_at))
                    >= _RESERVATION_REFRESH_SECONDS
                ]
                if stale:
                    reservations = self._reserve_remote(push.spec)
                stage_ms["reserve"] += (time.monotonic() - stage_started_at) * 1000
                for shard in reservations:
                    if shard.get("cached", False) or shard.get("cancelled", False):
                        continue
                    if not shard.get("write", True):
                        continue
                    if push.ready_event is not None and id(push) not in synchronized:
                        stage_started_at = time.monotonic()
                        push.ready_event.synchronize()
                        stage_ms["cuda"] += (time.monotonic() - stage_started_at) * 1000
                        synchronized.add(id(push))
                    if int(shard["nbytes"]) != push.tensor.nbytes:
                        raise RuntimeError(
                            "Reserved EC size does not match tensor for "
                            f"mm_hash={push.spec.mm_hash}"
                        )
                    ready.append((push, shard))
                    notifications.append((push, shard))
            if not ready and not notifications:
                return

            if ready:
                # One source per push: a sharded consumer reads the same bytes
                # into each of its ranks, so staging and registration happen
                # once however many destinations there are.
                unique: list[_PendingPush] = []
                source_index: dict[int, int] = {}
                for push, _ in ready:
                    if id(push) not in source_index:
                        source_index[id(push)] = len(unique)
                        unique.append(push)
                tensors = [push.tensor for push in unique]
                lengths = [tensor.nbytes for tensor in tensors]
                stage_started_at = time.monotonic()
                staged = self._stage_push_sources(tensors)
                registered_sources: list[int] = []
                staged_regions: list[tuple[int, int]] = []
                if staged is not None:
                    sources, staged_regions = staged
                    # The NIC reads outside the CUDA stream, so the staging
                    # copies have to have landed before the transfer starts.
                    if sources and sources[0].device.type == "cuda":
                        torch.accelerator.current_stream(
                            sources[0].device
                        ).synchronize()
                else:
                    sources = tensors
                    registered_sources = self._transfer.acquire_sources(tensors)
                addresses = [tensor.data_ptr() for tensor in sources]
                stage_ms["register"] = (time.monotonic() - stage_started_at) * 1000
                try:
                    by_session: dict[str, list[tuple[int, int]]] = {}
                    for push, shard in ready:
                        by_session.setdefault(str(shard["dst_session"]), []).append(
                            (source_index[id(push)], int(shard["dst_ptr"]))
                        )
                    stage_started_at = time.monotonic()

                    def write(session: str, items: list[tuple[int, int]]) -> None:
                        self._transfer.write(
                            session,
                            [addresses[index] for index, _ in items],
                            [dst for _, dst in items],
                            [lengths[index] for index, _ in items],
                        )

                    sessions = list(by_session.items())
                    # Shards are written concurrently: serialising them would
                    # make the transfer cost the sum of the ranks instead of
                    # the slowest one.
                    extra = [
                        self._shard_executor().submit(write, session, items)
                        for session, items in sessions[1:]
                    ]
                    try:
                        write(*sessions[0])
                    finally:
                        for future in extra:
                            future.result()
                    stage_ms["rdma"] = (time.monotonic() - stage_started_at) * 1000
                finally:
                    stage_started_at = time.monotonic()
                    self._release_push_staging(staged_regions)
                    self._transfer.release_sources(registered_sources)
                    stage_ms["unregister"] = (
                        time.monotonic() - stage_started_at
                    ) * 1000

            stage_started_at = time.monotonic()
            self._notify_completions(notifications)
            stage_ms["complete"] = (time.monotonic() - stage_started_at) * 1000
        except Exception:
            # A failed batch must not take the engine down with it: the
            # consumer is told to drop its reservations and this item falls
            # back to whatever the consumer can still do (pull, or a local
            # re-encode). Raising here would surface in
            # `build_connector_worker_meta` as a fatal EngineCore error.
            failed = True
            logger.exception(
                "EC Mooncake push batch failed for mm_hashes=%s",
                [push.spec.mm_hash for push in pushes],
            )
            self._abandon_pushes(pushes)
        finally:
            with self._active_push_sources_lock:
                for push in pushes:
                    key = (push.spec.mm_hash, id(push.tensor))
                    self._active_push_sources[key] -= 1
                    if self._active_push_sources[key] == 0:
                        del self._active_push_sources[key]
            stage_ms["total"] = (time.monotonic() - started_at) * 1000
            self._record_push_perf(
                stage_ms,
                stage_max_ms={"queue": max(queue_waits_ms, default=0.0)},
                item_count=len(pushes),
                # `ready` holds one entry per destination shard, so count the
                # distinct items rather than the writes.
                byte_count=sum(
                    push.tensor.nbytes for push in {id(p): p for p, _ in ready}.values()
                ),
                skipped_items=len(pushes) - len({id(push) for push, _ in ready}),
                failed=failed,
            )

    def _notify_completions(
        self, notifications: list[tuple[_PendingPush, dict[str, Any]]]
    ) -> None:
        """Tell the consumer, in one message per destination, what landed."""
        if not notifications:
            return
        by_destination: dict[str, list[tuple[_PendingPush, dict[str, Any]]]] = {}
        for push, reservation in notifications:
            by_destination.setdefault(
                str(reservation.get("addr", push.spec.consumer_zmq)), []
            ).append((push, reservation))
        for consumer_zmq, items in by_destination.items():
            result = self._control_client.request(
                consumer_zmq,
                {
                    "op": "complete_batch",
                    "items": [
                        {
                            "transfer_id": push.spec.transfer_id,
                            "reservation_id": reservation["reservation_id"],
                        }
                        for push, reservation in items
                    ],
                },
            )
            completions = result.get("items", []) if isinstance(result, dict) else []
            if len(completions) != len(items):
                raise RuntimeError("Malformed EC completion response")
            for (push, _), completion in zip(items, completions):
                if not completion.get("completed"):
                    raise RuntimeError(
                        f"Unknown EC reservation for mm_hash={push.spec.mm_hash}"
                    )

    def _abandon_pushes(self, pushes: list[_PendingPush]) -> None:
        """Release the consumer-side reservations of a batch that failed."""
        for push in pushes:
            shards: list[dict[str, Any]] = []
            if push.reservation.done() and not push.reservation.cancelled():
                with suppress(Exception):
                    shards = push.reservation.result()
            if not shards:
                shards = [{"addr": push.spec.consumer_zmq, "reservation_id": ""}]
            for shard in shards:
                with suppress(Exception):
                    self._control_client.request(
                        str(shard.get("addr", push.spec.consumer_zmq)),
                        {
                            "op": "cancel",
                            "transfer_id": push.spec.transfer_id,
                            "reservation_id": str(shard.get("reservation_id", "")),
                            "abandon": True,
                        },
                    )

    def _record_push_perf(
        self,
        stage_ms: dict[str, float],
        *,
        stage_max_ms: dict[str, float],
        item_count: int,
        byte_count: int,
        skipped_items: int,
        failed: bool,
    ) -> None:
        now = time.monotonic()
        report: tuple[_PushPerfWindow, int, int] | None = None
        with self._push_perf_lock:
            self._active_transfer_batches -= 1
            perf = self._push_perf
            perf.batches += 1
            perf.items += item_count
            perf.bytes += byte_count
            perf.skipped_items += skipped_items
            perf.failures += int(failed)
            for stage, elapsed_ms in stage_ms.items():
                perf.stage_totals_ms[stage] = (
                    perf.stage_totals_ms.get(stage, 0.0) + elapsed_ms
                )
                perf.stage_max_ms[stage] = max(
                    perf.stage_max_ms.get(stage, 0.0),
                    stage_max_ms.get(stage, elapsed_ms),
                )
            if (
                self._transfer_metrics_log_interval > 0
                and now - perf.started_at >= self._transfer_metrics_log_interval
            ):
                report = (
                    perf,
                    self._active_transfer_batches,
                    self._queued_transfer_batches,
                )
                self._push_perf = _PushPerfWindow(started_at=now)
        if report is None:
            return
        perf, active_batches, queued_batches = report
        batches = max(perf.batches, 1)
        items = max(perf.items, 1)
        stage_parts = []
        for stage in (
            "queue",
            "reserve",
            "cuda",
            "register",
            "rdma",
            "unregister",
            "complete",
            "total",
        ):
            divisor = items if stage == "queue" else batches
            average = perf.stage_totals_ms.get(stage, 0.0) / divisor
            maximum = perf.stage_max_ms.get(stage, 0.0)
            stage_parts.append(f"{stage}_ms={average:.1f}/{maximum:.1f}")
        stage_summary = " ".join(stage_parts)
        producer_metrics = dict(self._producer_metrics)
        self._producer_metrics.clear()
        logger.info(
            "EC Mooncake push perf: batches=%d items=%d bytes=%d "
            "batch_items=%.1f skipped=%d failures=%d active=%d queued=%d "
            "producer=%s queue_item_avg/max and stage_batch_avg/max: %s",
            perf.batches,
            perf.items,
            perf.bytes,
            perf.items / batches,
            perf.skipped_items,
            perf.failures,
            active_batches,
            queued_batches,
            producer_metrics,
            stage_summary,
        )

    def _flush_pending_pushes(self) -> None:
        if not self._pending_pushes:
            return
        grouped: dict[str, list[_PendingPush]] = {}
        for push in self._pending_pushes:
            grouped.setdefault(push.spec.consumer_zmq, []).append(push)
        self._pending_pushes = []
        for pushes in grouped.values():
            with self._push_perf_lock:
                self._queued_transfer_batches += 1
            future = self._io_executor.submit(self._push_batch, pushes)
            hashes = ",".join(push.spec.mm_hash for push in pushes)
            self._pending_saves.append((hashes, future))

    def _submit_push(
        self,
        tensor: torch.Tensor,
        spec: ECMooncakePushSpec,
        reservation: Future[list[dict[str, Any]]],
    ) -> None:
        ready_event = None
        if tensor.device.type == "cuda":
            ready_event = torch.Event()
            ready_event.record(torch.accelerator.current_stream(tensor.device))
        self._pending_pushes.append(
            _PendingPush(
                tensor=tensor,
                spec=spec,
                reservation=reservation,
                ready_event=ready_event,
                enqueued_at=time.monotonic(),
            )
        )

    def _submit_reserved_pushes(self, tensor: torch.Tensor, mm_hash: str) -> None:
        reservations = self._pending_reservations.pop(mm_hash, deque())
        if reservations:
            with self._active_push_sources_lock:
                self._active_push_sources[(mm_hash, id(tensor))] += len(reservations)
        for spec, reservation in reservations:
            self._submit_push(tensor, spec, reservation)

    def _cancel_orphaned_reservation(
        self,
        spec: ECMooncakePushSpec,
        reservation: Future[list[dict[str, Any]]],
    ) -> None:
        try:
            for shard in reservation.result():
                if shard.get("cached", False) or shard.get("cancelled", False):
                    continue
                self._control_client.request(
                    str(shard.get("addr", spec.consumer_zmq)),
                    {
                        "op": "cancel",
                        "transfer_id": spec.transfer_id,
                        "reservation_id": str(shard.get("reservation_id", "")),
                        "abandon": True,
                    },
                )
        except Exception:
            logger.exception(
                "Failed to cancel orphaned EC reservation for transfer_id=%s",
                spec.transfer_id,
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if not self.is_producer:
            return None, None

        Reserved = tuple[ECMooncakePushSpec, Future[list[dict[str, Any]]]]
        orphaned: list[Reserved] = []
        for mm_hash, reservations in list(self._pending_reservations.items()):
            remaining: deque[Reserved] = deque()
            for spec, reservation in reservations:
                if spec.request_id in finished_req_ids:
                    orphaned.append((spec, reservation))
                else:
                    remaining.append((spec, reservation))
            if remaining:
                self._pending_reservations[mm_hash] = remaining
            else:
                self._pending_reservations.pop(mm_hash)

        for spec, reservation in orphaned:
            future = self._io_executor.submit(
                self._cancel_orphaned_reservation, spec, reservation
            )
            self._pending_saves.append((f"cancel:{spec.transfer_id}", future))
        return None, None

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        if not self.is_producer:
            return
        tensor = encoder_cache[mm_hash]
        if mm_hash in self._pending_reservations:
            self._submit_reserved_pushes(tensor, mm_hash)

    def build_connector_worker_meta(self) -> ECMooncakeWorkerMetadata | None:
        if self.is_consumer and not self._is_receiving_rank:
            # `loaded` is intersected across reporting ranks, so a stage that
            # never loads must not report at all rather than report nothing.
            return None

        self._flush_pending_pushes()
        saves = self._pending_saves
        completed_saves = []
        self._pending_saves = [
            (mm_hash, future) for mm_hash, future in saves if not future.done()
        ]
        for mm_hash, future in saves:
            if future.done():
                completed_saves.append((mm_hash, future))
        for mm_hash, future in completed_saves:
            try:
                future.result()
            except Exception:
                # Publishing is best-effort: a consumer that cannot fetch this
                # item falls back to encoding it locally. Failing the step
                # instead would take the whole engine down.
                self._producer_metrics["saves_failed"] += 1
                logger.exception(
                    "EC Mooncake async save failed for mm_hash=%s", mm_hash
                )
        with self._consumer_lock:
            reclaimed = self._consumer_reclaimed
            self._consumer_reclaimed = set()
        meta = ECMooncakeWorkerMetadata(
            loaded=self._completed_loads,
            failed_loads=self._failed_loads,
            reclaimed=reclaimed,
            pending_loads=False,
            pending_saves=bool(self._pending_saves),
        )
        self._completed_loads = set()
        self._failed_loads = set()
        if self.is_consumer:
            self._maybe_log_consumer_worker_metrics()
        return meta

    def close(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._flush_pending_pushes()
        self._io_executor.shutdown(wait=True, cancel_futures=True)
        if self._shard_pool is not None:
            self._shard_pool.shutdown(wait=True, cancel_futures=True)
        self._control_executor.shutdown(wait=True, cancel_futures=True)
        # Every thread that could hold a control socket is stopped by now.
        self._control_client.close()
        if self._control_server is not None:
            self._control_server.close()
        if self._consumer_pool is not None and self._transfer.unregister_memory(
            self._consumer_pool
        ):
            self._consumer_pool = None
            self._consumer_pool_allocator = None
            self._consumer_residents.clear()
            self._consumer_retire_events.clear()
            self._consumer_pending_frees.clear()
        self._transfer.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
