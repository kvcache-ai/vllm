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
from collections.abc import Callable, Collection
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
    ECConnectorWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    _get_encoder_cache_hidden_dim,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
    ECMooncakeSchedulerConfig,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

__all__ = [
    "ECMooncakeConnector",
    "ECMooncakeConnectorMetadata",
    "ECMooncakeLoadSpec",
    "ECMooncakePushSpec",
    "ECMooncakeWorkerMetadata",
]

logger = init_logger(__name__)

_T = TypeVar("_T")

_LEASE_TTL_SECONDS = 300
_RESERVATION_REFRESH_SECONDS = _LEASE_TTL_SECONDS / 2
_RESERVATION_REAP_INTERVAL_SECONDS = 1
_DRAIN_MIN_INTERVAL = 0.005
# Readiness notifications are advisory: the scheduler also learns from the
# reserve reply. Cap the queue so a shard nobody subscribed to cannot grow
# without bound.
_MAX_PENDING_EVENTS = 4096
# A cancelled transfer stays on the scheduler's ignore list for as long as the
# worker refuses to reserve it again. The count is a backstop for a rate that
# outruns that TTL; the race it guards is a single drain interval wide.
_MAX_CANCELLED_TRANSFER_IDS = 1 << 16

_MOONCAKE_IMPORT_ERROR: ImportError | None
try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    TransferEngine = None  # type: ignore[misc, assignment]
    _MOONCAKE_IMPORT_ERROR = e
else:
    _MOONCAKE_IMPORT_ERROR = None


@dataclass
class _PushSourceRegistration:
    tensor: torch.Tensor
    nbytes: int
    users: int = 1


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


@dataclass(frozen=True)
class _PushCompletion:
    accepted: bool
    became_ready: bool = False


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


class _ControlChannel:
    """Reusable REQ sockets for the ZMQ control plane.

    One context and one connection per message costs a thread spawn plus a
    TCP handshake, and the push path sends one message per reserve, complete
    and cancel. Sockets are cached per thread because a REQ socket is neither
    thread-safe nor usable after a failed exchange.
    """

    def __init__(self, timeout_ms: int):
        self._context = zmq.Context()
        self._timeout_ms = timeout_ms
        self._local = threading.local()

    def _sockets(self) -> dict[str, zmq.Socket]:
        sockets = getattr(self._local, "sockets", None)
        if sockets is None:
            sockets = {}
            self._local.sockets = sockets
        return sockets

    def _discard(self, addr: str) -> None:
        socket = self._sockets().pop(addr, None)
        if socket is not None:
            socket.close(linger=0)

    def send(self, addr: str, payload: dict[str, Any]) -> dict[str, Any]:
        sockets = self._sockets()
        socket = sockets.get(addr)
        if socket is None:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(addr)
            sockets[addr] = socket
        try:
            socket.send_json(payload)
            response = socket.recv_json()
        except Exception:
            # A REQ socket cannot recover from a half-finished exchange.
            self._discard(addr)
            raise
        assert isinstance(response, dict)
        return response

    def request(self, addr: str, payload: dict[str, Any]) -> Any:
        response = self.send(addr, payload)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "EC control request failed"))
        return response.get("result")

    def close(self) -> None:
        # Callers must have stopped every thread that used this channel.
        self._context.destroy(linger=0)


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


class ECMooncakeControlServer:
    """Expose consumer reservations over a lightweight ZMQ control channel."""

    def __init__(
        self,
        host: str,
        port: int,
        reserve: Callable[[dict[str, Any]], dict[str, Any]],
        status: Callable[[str], dict[str, Any] | None],
        complete: Callable[[str, str], _PushCompletion],
        cancel: Callable[[str, str, bool], bool],
        reap: Callable[[], int],
        metrics_log_interval: float = 10,
        peer_ports: list[int] | None = None,
        device: torch.device | None = None,
    ):
        self.host = host
        self.port = port
        self.peer_ports = peer_ports or [port]
        self._device = device
        self.event_port: int | None = None
        self._reserve = reserve
        self._status = status
        self._complete = complete
        self._cancel = cancel
        self._reap = reap
        self._metrics_log_interval = metrics_log_interval
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: Exception | None = None

    def start(self) -> None:
        def loop() -> None:
            if self._device is not None and self._device.type == "cuda":
                # Reserving can retire an entry, and the event that orders its
                # reuse is created on the recording thread's device rather
                # than the stream's. A thread starts on device 0, which under
                # a shard-local CUDA_VISIBLE_DEVICES is a peer's GPU, so
                # without this every shard but the first strands a primary
                # context there. The event orders correctly either way; what
                # it costs is a few hundred MiB on someone else's card.
                torch.accelerator.set_device_index(self._device.index or 0)
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            event_socket = context.socket(zmq.PUSH)
            pending_events: deque[dict[str, Any]] = deque()
            metrics: Counter[str] = Counter()

            def queue_event(event: dict[str, Any]) -> None:
                # The shard tag lets the scheduler tell each rank's readiness
                # apart; a transfer is only loadable once every rank has it.
                event["shard"] = self.port
                if len(pending_events) >= _MAX_PENDING_EVENTS:
                    pending_events.popleft()
                    metrics["events_dropped"] += 1
                pending_events.append(event)
                metrics["events_queued"] += 1

            metrics_started_at = time.monotonic()
            last_reap_at = metrics_started_at
            socket.setsockopt(zmq.RCVTIMEO, 100)
            try:
                socket.bind(f"tcp://{self.host}:{self.port}")
                self.event_port = event_socket.bind_to_random_port(f"tcp://{self.host}")
            except Exception as e:
                self._startup_error = e
                self._started.set()
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()
                return
            self._started.set()
            try:
                while not self._stop.is_set():
                    while pending_events:
                        try:
                            event_socket.send_json(
                                pending_events[0], flags=zmq.DONTWAIT
                            )
                        except zmq.Again:
                            break
                        pending_events.popleft()
                        metrics["events_sent"] += 1
                    now = time.monotonic()
                    if now - last_reap_at >= _RESERVATION_REAP_INTERVAL_SECONDS:
                        metrics["reservations_reaped"] += self._reap()
                        last_reap_at = now
                    if (
                        self._metrics_log_interval > 0
                        and now - metrics_started_at >= self._metrics_log_interval
                    ):
                        logger.info(
                            "EC Mooncake consumer control: requests=%s, "
                            "events_queued=%d, events_sent=%d, events_dropped=%d, "
                            "event_backlog=%d, reservations_reaped=%d",
                            {
                                key.removeprefix("request_"): value
                                for key, value in metrics.items()
                                if key.startswith("request_")
                            },
                            metrics["events_queued"],
                            metrics["events_sent"],
                            metrics["events_dropped"],
                            len(pending_events),
                            metrics["reservations_reaped"],
                        )
                        metrics.clear()
                        metrics_started_at = now
                    try:
                        request = socket.recv_json()
                    except zmq.Again:
                        continue
                    try:
                        op = request.get("op")
                        result: Any = None
                        metrics[f"request_{op}"] += 1
                        if op == "reserve":
                            result = self._reserve(request)
                            if result.get("ready"):
                                transfer_id = str(request["transfer_id"])
                                status = self._status(transfer_id)
                                if status is not None:
                                    queue_event({"transfer_id": transfer_id, **status})
                        elif op == "status":
                            result = self._status(str(request["transfer_id"]))
                        elif op == "event_port":
                            result = self.event_port
                        elif op == "peers":
                            # Every consumer shard receives its own copy, so a
                            # producer holding one address needs the rest.
                            result = {"ports": self.peer_ports}
                        elif op in ("complete", "complete_batch"):
                            items = (
                                request["items"]
                                if op == "complete_batch"
                                else [request]
                            )
                            completions = []
                            for item in items:
                                transfer_id = str(item["transfer_id"])
                                completion = self._complete(
                                    transfer_id,
                                    str(item["reservation_id"]),
                                )
                                completions.append(
                                    {
                                        "completed": completion.accepted,
                                        "became_ready": completion.became_ready,
                                    }
                                )
                                if not completion.became_ready:
                                    continue
                                status = self._status(transfer_id)
                                if status is not None:
                                    queue_event({"transfer_id": transfer_id, **status})
                            result = (
                                {"items": completions}
                                if op == "complete_batch"
                                else completions[0]
                            )
                        elif op == "cancel":
                            result = {
                                "cancelled": self._cancel(
                                    str(request["transfer_id"]),
                                    str(request.get("reservation_id", "")),
                                    bool(request.get("abandon", False)),
                                )
                            }
                        else:
                            raise ValueError(f"unknown control op: {op!r}")
                        socket.send_json({"ok": True, "result": result})
                    except Exception as e:
                        socket.send_json({"ok": False, "error": str(e)})
            finally:
                socket.close(linger=0)
                event_socket.close(linger=0)
                context.term()

        self._thread = threading.Thread(
            target=loop, name="ec-mooncake-control", daemon=True
        )
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("EC Mooncake control channel failed to start")
        if self._startup_error is not None:
            raise RuntimeError("EC Mooncake control channel failed to bind") from (
                self._startup_error
            )
        logger.info(
            "EC Mooncake control channel listening on tcp://%s:%d (events tcp://%s:%d)",
            self.host,
            self.port,
            self.host,
            self.event_port,
        )

    def shutdown(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()


class ECMooncakeConnector(ECConnectorBase):
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

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._scheduler: ECMooncakeScheduler | None = None
        if _MOONCAKE_IMPORT_ERROR is not None or TransferEngine is None:
            raise ImportError(
                "Install mooncake-transfer-engine (see "
                "https://github.com/kvcache-ai/Mooncake ) to use ECMooncakeConnector."
            ) from _MOONCAKE_IMPORT_ERROR

        parallel_config = vllm_config.parallel_config
        ec_cfg_early = vllm_config.ec_transfer_config
        assert ec_cfg_early is not None
        if ec_cfg_early.is_ec_producer:
            # The producer holds one copy of each encoder output and addresses
            # consumers directly; sharding or replicating it would only
            # duplicate the push.
            if parallel_config.tensor_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers require tensor_parallel_size=1."
                )
            if parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers do not support pipeline parallelism."
                )
            if parallel_config.data_parallel_size > 1:
                raise ValueError(
                    "ECMooncakeConnector producers require data_parallel_size=1."
                )

        # Each data-parallel replica runs its own scheduler and its own control
        # channels, so their ports must not overlap. `data_parallel_index` is the
        # only field that identifies the replica in both cases: a non-MoE replica
        # is reconfigured to look like DP=1, which resets `data_parallel_rank`
        # and `data_parallel_size`. Deriving the offset from the config rather
        # than from a process group keeps the scheduler, which has no groups, in
        # agreement with its workers.
        self._control_port_offset = (
            parallel_config.data_parallel_index * parallel_config.tensor_parallel_size
        )

        self._role = role
        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None
        self._ec_cfg = ec_cfg
        self._extra = self._ec_cfg.ec_connector_extra_config
        self._protocol: str = self._extra.get("mooncake_protocol", "rdma")
        reservation_port = self._extra.get("reservation_zmq_port")
        self._reservation_zmq_port = (
            int(reservation_port) if reservation_port is not None else None
        )
        self._reservation_zmq_addr: str | None = self._extra.get("reservation_zmq_addr")
        if (
            self._reservation_zmq_addr is None
            and self._reservation_zmq_port is not None
        ):
            base = self._reservation_zmq_port + self._control_port_offset
            self._reservation_zmq_addr = f"tcp://127.0.0.1:{base}"
        self._registered_capacity = int(self._ec_cfg.ec_buffer_size)
        if self._registered_capacity <= 0:
            raise ValueError("ECMooncakeConnector requires ec_buffer_size > 0.")
        self._model_config = vllm_config.model_config

        pool_size = self._extra.get(
            "consumer_buffer_pool_size", self._registered_capacity
        )
        self._consumer_pool_capacity = int(pool_size)
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
        self._consumer_pool_disabled = self._consumer_pool_capacity <= 0
        self._consumer_lock = threading.Lock()
        self._push_reservations: dict[str, _PushReservation] = {}
        self._cancelled_transfers: OrderedDict[str, float] = OrderedDict()
        self._control_server: ECMooncakeControlServer | None = None
        self._consumer_metrics_log_interval = float(
            self._extra.get("consumer_metrics_log_interval", 10)
        )
        self._consumer_metrics_started_at = time.monotonic()
        self._consumer_worker_metrics: Counter[str] = Counter()
        self._active_push_sources: Counter[tuple[str, int]] = Counter()
        self._active_push_sources_lock = threading.Lock()

        # Worker producer
        self._engine: TransferEngine | None = None
        self._engine_lock = threading.Lock()
        self._hostname = get_ip()
        # Published encoder outputs, referenced while a pull is reading them.
        self._pending_unregister: dict[int, torch.Tensor] = {}
        self._push_source_registrations: dict[int, _PushSourceRegistration] = {}
        self._push_source_registration_lock = threading.Lock()
        producer_pool = self._extra.get(
            "producer_buffer_pool_size", self._registered_capacity
        )
        self._producer_pool_capacity = int(producer_pool)
        self._producer_pool: torch.Tensor | None = None
        self._producer_pool_allocator: _ContiguousAllocator | None = None
        self._producer_pool_disabled = self._producer_pool_capacity <= 0
        self._producer_pool_lock = threading.Lock()
        transfer_workers = int(self._extra.get("transfer_max_workers", 4))
        control_workers = int(self._extra.get("control_max_workers", 8))
        self._transfer_metrics_log_interval = float(
            self._extra.get("transfer_metrics_log_interval", 10)
        )
        self._control_channel = _ControlChannel(
            int(float(self._extra.get("control_timeout_s", 30)) * 1000)
        )
        self._producer_metrics: Counter[str] = Counter()
        self._io_executor = ThreadPoolExecutor(
            max_workers=transfer_workers, thread_name_prefix="ec-mooncake-transfer"
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=control_workers, thread_name_prefix="ec-mooncake-control"
        )
        self._consumer_shard_cache: dict[str, list[str]] = {}
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

        if (
            role == ECConnectorRole.SCHEDULER
            and self.is_consumer
            and not self._reservation_zmq_addr
        ):
            raise ValueError(
                "ec_consumer with ECMooncakeConnector requires "
                "reservation_zmq_port or reservation_zmq_addr."
            )
        if role == ECConnectorRole.SCHEDULER:
            encoder_cache_hidden_dim = (
                _get_encoder_cache_hidden_dim(vllm_config) if self.is_producer else None
            )
            self._scheduler = ECMooncakeScheduler(
                ECMooncakeSchedulerConfig(
                    is_producer=self.is_producer,
                    is_consumer=self.is_consumer,
                    reservation_zmq_addr=self._reservation_zmq_addr,
                    consumer_pool_capacity=self._consumer_pool_capacity,
                    push_wait_timeout=float(self._extra.get("push_wait_timeout_s", 60)),
                    consumer_metrics_log_interval=(self._consumer_metrics_log_interval),
                    encoder_cache_hidden_dim=encoder_cache_hidden_dim,
                ),
                model_config=self._model_config,
                control_request=self._control_channel.request,
                submit_control=self._control_executor.submit,
            )

    def _ensure_engine(self) -> TransferEngine:
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            eng = TransferEngine()
            ret = eng.initialize(self._hostname, "P2PHANDSHAKE", self._protocol, "")
            if ret != 0:
                raise RuntimeError("Mooncake TransferEngine initialization failed.")
            self._engine = eng
            logger.info(
                "ECMooncakeConnector TransferEngine ready at %s:%d",
                self._hostname,
                eng.get_rpc_port(),
            )
        return self._engine

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

    def start_worker_services(self) -> None:
        if (
            self._role != ECConnectorRole.WORKER
            or not self.is_consumer
            or self._reservation_zmq_port is None
            or self._control_server is not None
        ):
            return
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Later pipeline stages hold no encoder outputs, so they need
            # neither a receive pool nor a control channel.
            return
        raw_device = self._ec_cfg.ec_buffer_device
        device_name = (
            raw_device.lower() if isinstance(raw_device, str) and raw_device else "cuda"
        )
        self._ensure_consumer_pool(torch.device(device_name), allow_host=True)
        if self._consumer_pool is None:
            raise RuntimeError(
                "Mooncake push mode requires a registered consumer buffer pool."
            )
        base_port = self._reservation_zmq_port + self._control_port_offset
        self._control_server = ECMooncakeControlServer(
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
        self._control_server.start()

    def _unregister_memory(self, tensor: torch.Tensor) -> bool:
        assert self._engine is not None
        ret = self._engine.unregister_memory(tensor.data_ptr())
        if ret != 0:
            logger.error(
                "Mooncake EC memory unregistration failed for address %d: %d",
                tensor.data_ptr(),
                ret,
            )
            self._pending_unregister[tensor.data_ptr()] = tensor
            return False
        self._pending_unregister.pop(tensor.data_ptr(), None)
        return True

    def _unregister_memories(self, tensors: list[torch.Tensor]) -> None:
        assert self._engine is not None
        addresses = [tensor.data_ptr() for tensor in tensors]
        ret = self._engine.batch_unregister_memory(addresses)
        if ret != 0:
            for tensor in tensors:
                self._pending_unregister[tensor.data_ptr()] = tensor
            logger.warning(
                "Keeping %d EC tensors alive after Mooncake unregistration failure",
                len(tensors),
            )
            return
        for address in addresses:
            self._pending_unregister.pop(address, None)

    @staticmethod
    def _push_source_range(tensor: torch.Tensor) -> tuple[int, int]:
        # Register exactly the bytes that will be transferred. One encoder
        # batch returns its items as views of a single storage (models split
        # the batched embeddings, e.g. `image_embeds.split(sizes)`), so
        # registering the whole storage would overlap the per-tensor
        # registration a sibling item takes -- and Mooncake rejects
        # overlapping memory regions.
        return tensor.data_ptr(), tensor.nbytes

    def _acquire_push_source_registrations(
        self, tensors: list[torch.Tensor]
    ) -> list[int]:
        ranges: dict[int, tuple[int, torch.Tensor]] = {}
        for tensor in tensors:
            address, nbytes = self._push_source_range(tensor)
            ranges.setdefault(address, (nbytes, tensor))

        eng = self._ensure_engine()
        acquired: list[int] = []
        new_addresses: list[int] = []
        new_lengths: list[int] = []
        with self._push_source_registration_lock:
            for address, (nbytes, tensor) in ranges.items():
                entry = self._push_source_registrations.get(address)
                if entry is not None:
                    if entry.nbytes != nbytes:
                        raise RuntimeError(
                            "Mooncake EC source storage changed size while registered"
                        )
                    entry.users += 1
                    acquired.append(address)
                    continue
                new_addresses.append(address)
                new_lengths.append(nbytes)
                self._push_source_registrations[address] = _PushSourceRegistration(
                    tensor=tensor,
                    nbytes=nbytes,
                )
                acquired.append(address)

            if new_addresses:
                ret = eng.batch_register_memory(new_addresses, new_lengths)
                if ret != 0:
                    for address in acquired:
                        entry = self._push_source_registrations[address]
                        entry.users -= 1
                        if entry.users == 0:
                            del self._push_source_registrations[address]
                    raise RuntimeError("Mooncake EC source registration failed")
        return acquired

    def _release_push_source_registrations(self, addresses: list[int]) -> bool:
        if not addresses:
            return True
        with self._push_source_registration_lock:
            unused = []
            for address in addresses:
                entry = self._push_source_registrations.get(address)
                if entry is None:
                    continue
                entry.users -= 1
                if entry.users == 0:
                    unused.append(address)
            if not unused:
                return True
            ret = self._ensure_engine().batch_unregister_memory(unused)
            if ret != 0:
                logger.warning(
                    "Keeping %d EC source tensors registered after Mooncake "
                    "unregistration failure",
                    len(unused),
                )
                return False
            for address in unused:
                del self._push_source_registrations[address]
                self._pending_unregister.pop(address, None)
            return True

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
                ret = self._ensure_engine().batch_register_memory(
                    [pool.data_ptr()], [pool.nbytes]
                )
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
                ret = self._ensure_engine().batch_register_memory(
                    [pool.data_ptr()], [pool.nbytes]
                )
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

    def take_unavailable_requests(self) -> set[str]:
        assert self._scheduler is not None
        return self._scheduler.take_unavailable_requests()

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

        eng = self._ensure_engine()
        return {
            "reservation_id": reservation.reservation_id,
            "dst_session": f"{self._hostname}:{eng.get_rpc_port()}",
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

    def _complete_push(self, transfer_id: str, reservation_id: str) -> _PushCompletion:
        with self._consumer_lock:
            reservation = self._push_reservations.get(transfer_id)
            if reservation is None or reservation.reservation_id != reservation_id:
                self._consumer_worker_metrics["completions_rejected"] += 1
                return _PushCompletion(False)
            if reservation.ready:
                self._consumer_worker_metrics["completions_repeated"] += 1
                return _PushCompletion(True)
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
                return _PushCompletion(True)
            reservation.ready = True
            reservation.expires_at = time.monotonic() + _LEASE_TTL_SECONDS
            return _PushCompletion(True, became_ready=True)

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

    def _send_control(self, addr: str, request: dict[str, Any]) -> Any:
        return self._control_channel.request(addr, request)

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

    def _consumer_shards(self, base_addr: str) -> list[str]:
        """Every control channel of the consumer reachable at `base_addr`.

        A tensor-parallel consumer gathers from each rank's own cache, so each
        rank receives its own copy. Asking the first one for the roster keeps
        the address list out of the request and the proxy configuration.
        """
        cached = self._consumer_shard_cache.get(base_addr)
        if cached is not None:
            return cached
        shards = [base_addr]
        try:
            reply = self._send_control(base_addr, {"op": "peers"})
            ports = reply.get("ports") if isinstance(reply, dict) else None
            if ports:
                prefix = base_addr.rsplit(":", 1)[0]
                shards = [f"{prefix}:{int(port)}" for port in ports]
        except Exception:
            # An older consumer does not answer this, and it can only be
            # unsharded, so its single address is the whole roster.
            logger.warning(
                "EC Mooncake consumer at %s did not report its shards; "
                "assuming it is unsharded.",
                base_addr,
                exc_info=True,
            )
        self._consumer_shard_cache[base_addr] = shards
        if len(shards) > 1:
            logger.info(
                "EC Mooncake consumer at %s has %d shards", base_addr, len(shards)
            )
        return shards

    def _reserve_one(self, addr: str, spec: ECMooncakePushSpec) -> dict[str, Any]:
        result = self._send_control(
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
        shards = self._consumer_shards(spec.consumer_zmq)
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
        for addr in self._consumer_shards(consumer_zmq):
            result = self._send_control(
                addr,
                {
                    "op": "cancel",
                    "transfer_id": transfer_id,
                    "reservation_id": reservation_id,
                },
            )
            cancelled |= isinstance(result, dict) and bool(result.get("cancelled"))
        return cancelled

    def start_save_caches(self, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        for spec in metadata.pushes:
            reservation = self._control_executor.submit(self._reserve_remote, spec)
            self._pending_reservations.setdefault(spec.mm_hash, deque()).append(
                (spec, reservation)
            )
        encoder_cache = kwargs.get("encoder_cache")
        if not isinstance(encoder_cache, dict):
            return
        for mm_hash in dict.fromkeys(spec.mm_hash for spec in metadata.pushes):
            tensor = encoder_cache.get(mm_hash)
            if tensor is not None:
                self._submit_reserved_pushes(tensor, mm_hash)

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        self._resolve_consumer_rank()
        if not self._is_receiving_rank:
            # Reached on steps with no work, from a stage that never gathers
            # multimodal embeddings. Taking a transfer here would fail for
            # want of a reservation and fail the load for everyone.
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        self._ensure_engine()
        raw_buf = self._ec_cfg.ec_buffer_device
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
                eng = self._ensure_engine()
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
                    registered_sources = self._acquire_push_source_registrations(
                        tensors
                    )
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
                        ret = eng.batch_transfer_sync_write(
                            session,
                            [addresses[index] for index, _ in items],
                            [dst for _, dst in items],
                            [lengths[index] for index, _ in items],
                        )
                        if ret != 0:
                            raise RuntimeError(
                                f"Mooncake EC push to {session} failed with "
                                f"status {ret}"
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
                    self._release_push_source_registrations(registered_sources)
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
            result = self._send_control(
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
                    self._send_control(
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
                self._send_control(
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
        if not self.is_producer or self._role != ECConnectorRole.WORKER:
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
        if not self.is_producer or self._role != ECConnectorRole.WORKER:
            return
        tensor = encoder_cache[mm_hash]
        if mm_hash in self._pending_reservations:
            self._submit_reserved_pushes(tensor, mm_hash)

    def has_cache_item(self, identifier: str) -> bool:
        assert self._scheduler is not None
        return self._scheduler.has_cache_item(identifier)

    def ensure_cache_available(
        self,
        request: Any,
        num_computed_tokens: int,
        local_cache_hashes: Collection[str] | None = None,
    ) -> bool:
        assert self._scheduler is not None
        return self._scheduler.ensure_cache_available(
            request, num_computed_tokens, local_cache_hashes
        )

    def update_state_after_alloc(self, request: Any, index: int) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(request, index)

    def update_state_after_free(self, request: Any, index: int) -> None:
        assert self._scheduler is not None
        self._scheduler.update_state_after_free(request, index)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def build_connector_worker_meta(self) -> ECConnectorWorkerMetadata | None:
        if self._role != ECConnectorRole.WORKER:
            return None
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

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        assert self._scheduler is not None
        self._scheduler.update_connector_output(connector_output)

    def has_pending_push_work(self) -> bool:
        assert self._scheduler is not None
        return self._scheduler.has_pending_push_work()

    def request_finished(self, request: Any) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._flush_pending_pushes()
        self._io_executor.shutdown(wait=True, cancel_futures=True)
        if self._shard_pool is not None:
            self._shard_pool.shutdown(wait=True, cancel_futures=True)
        self._control_executor.shutdown(wait=True, cancel_futures=True)
        # Every thread that could hold a control socket is stopped by now.
        self._control_channel.close()
        if self._control_server is not None:
            self._control_server.shutdown()
        if self._scheduler is not None:
            self._scheduler.close()

        if self._engine is not None:
            if self._consumer_pool is not None and self._unregister_memory(
                self._consumer_pool
            ):
                self._consumer_pool = None
                self._consumer_pool_allocator = None
                self._consumer_residents.clear()
                self._consumer_retire_events.clear()
                self._consumer_pending_frees.clear()
            # Published tensors and in-flight push sources share one refcounted
            # registration table, so a single pass covers both.
            with self._push_source_registration_lock:
                addresses = list(self._push_source_registrations)
                addresses.extend(self._pending_unregister)
                unregistered = True
                if addresses:
                    ret = self._engine.batch_unregister_memory(
                        list(dict.fromkeys(addresses))
                    )
                    if ret != 0:
                        unregistered = False
                        logger.error(
                            "Mooncake EC batch memory unregistration failed: %d", ret
                        )
                if unregistered:
                    self._push_source_registrations.clear()
                    self._pending_unregister.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.shutdown()
