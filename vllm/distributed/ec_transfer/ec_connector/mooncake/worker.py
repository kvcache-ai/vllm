# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Encoder-cache (EC) connector backed by Mooncake TransferEngine.

Used in disaggregated setups where an encoder / prefill instance produces
multimodal encoder outputs and a decode instance loads them over RDMA-capable
Mooncake transport instead of shared filesystem.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

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
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    MemoryAllocation,
    ProducerMemoryPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushManager,
    ProducerPushRecord,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    CancellationOutcome,
    ConsumerReservationManager,
    ConsumerReservationState,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_LEASE_TTL_SECONDS = 300
_RESERVATION_REFRESH_SECONDS = _LEASE_TTL_SECONDS / 2
_MAX_CANCELLED_TRANSFER_IDS = 1 << 16


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
        self._transfer = MooncakeTransfer(hostname, config.protocol)
        self._consumer_worker_metrics: Counter[str] = Counter()
        self._consumer_memory = ConsumerMemoryPool(
            config.consumer_pool_size,
            self._transfer,
        )
        self._reservations = ConsumerReservationManager(
            self._consumer_memory,
            _LEASE_TTL_SECONDS,
            _MAX_CANCELLED_TRANSFER_IDS,
        )
        self._consumer_rank_resolved = False
        self._is_receiving_rank = True
        self._tp_rank = 0
        self._tp_size = 1
        self._control_server: ConsumerControlServer | None = None
        self._consumer_metrics_log_interval = config.consumer_metrics_log_interval
        self._consumer_metrics_started_at = time.monotonic()
        # Worker producer
        self._producer_memory = ProducerMemoryPool(
            config.producer_pool_size,
            self._transfer,
        )
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
        self._producer_pushes = ProducerPushManager()
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
        self._consumer_memory.prepare(
            torch.device(device_name),
            receiving_rank=self._is_receiving_rank,
            allow_host=True,
        )
        consumer_pool = self._consumer_memory.tensor
        if consumer_pool is None:
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
            device=consumer_pool.device,
        )
        try:
            self._control_server.start()
        except Exception:
            self._control_server.close()
            self._control_server = None
            raise

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
        with self._consumer_memory.lock:
            reservations = self._reservations.active_records()
            ready = [
                record.mm_hash
                for record in reservations
                if record.state is ConsumerReservationState.READY
            ]
            pending = [
                record.mm_hash
                for record in reservations
                if record.state is not ConsumerReservationState.READY
            ]
            metrics = dict(self._consumer_worker_metrics)
            self._consumer_worker_metrics.clear()
            metrics.update(self._consumer_memory.take_metrics())
            residents, live, retired, pending_frees = self._consumer_memory.stats()
            oldest_reservation_ms = max(
                ((now - reservation.created_at) * 1000 for reservation in reservations),
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

    def _expire_push_reservations(self) -> int:
        return self._record_expiry_metrics(self._reservations.expire())

    def _record_expiry_metrics(self, counts: tuple[int, int, int]) -> int:
        expired, deferred, tombstones_dropped = counts
        self._consumer_worker_metrics["reservations_expired"] += expired
        self._consumer_worker_metrics["cancellations_deferred"] += deferred
        self._consumer_worker_metrics["cancel_records_dropped"] += tombstones_dropped
        return expired

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

        self._expire_push_reservations()
        reservation, should_write, reused, expiry_counts = self._reservations.reserve(
            transfer_id, mm_hash, nbytes, shape, dtype_name, dtype
        )
        self._record_expiry_metrics(expiry_counts)
        if reservation is None:
            raise RuntimeError("EC consumer buffer pool is full")
        if reservation.state in {
            ConsumerReservationState.CANCEL_PENDING,
            ConsumerReservationState.CANCELLED,
        }:
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
        if reused:
            key = (
                "reservations_reused_ready"
                if reservation.state is ConsumerReservationState.READY
                else "reservations_reused_pending"
            )
            self._consumer_worker_metrics[key] += 1
        elif reservation.lease is not None:
            self._consumer_worker_metrics["reservations_cached"] += 1
        else:
            self._consumer_worker_metrics["reservations_created"] += 1
        assert reservation.allocation is not None

        return {
            "reservation_id": reservation.reservation_id,
            "dst_session": self._transfer.local_session(),
            "dst_ptr": reservation.allocation.tensor.data_ptr(),
            "nbytes": reservation.allocation.tensor.nbytes,
            "write": should_write,
            "ready": reservation.state is ConsumerReservationState.READY,
            "cached": reservation.lease is not None,
        }

    def _push_status(self, transfer_id: str) -> dict[str, Any] | None:
        reservation = self._reservations.status(transfer_id)
        if reservation is None:
            return None
        assert reservation.allocation is not None
        return {
            "mm_hash": reservation.mm_hash,
            "ready": reservation.state is ConsumerReservationState.READY,
            "reservation_id": reservation.reservation_id,
            "nbytes": reservation.allocation.tensor.nbytes,
            "shape": list(reservation.shape),
            "dtype": reservation.dtype,
        }

    def _complete_push(
        self, transfer_id: str, reservation_id: str
    ) -> ControlCompletion:
        result = self._reservations.complete(transfer_id, reservation_id)
        if not result.accepted:
            self._consumer_worker_metrics["completions_rejected"] += 1
        elif result.repeated:
            self._consumer_worker_metrics["completions_repeated"] += 1
        else:
            self._consumer_worker_metrics["completions_accepted"] += 1
        if result.discarded:
            self._consumer_worker_metrics["reservations_discarded"] += 1
        return ControlCompletion(result.accepted, result.became_ready)

    def _cancel_push(
        self,
        transfer_id: str,
        reservation_id: str,
        abandon: bool = False,
        refresh: bool = False,
    ) -> bool:
        outcome, tombstones_dropped = self._reservations.cancel(
            transfer_id, reservation_id, abandon, refresh
        )
        metrics = {
            CancellationOutcome.REJECTED: "cancellations_rejected",
            CancellationOutcome.PRE_RESERVED: "cancellations_pre_reserved",
            CancellationOutcome.DEFERRED: "cancellations_deferred",
            CancellationOutcome.CANCELLED: "reservations_cancelled",
        }
        self._consumer_worker_metrics[metrics[outcome]] += 1
        self._consumer_worker_metrics["cancel_records_dropped"] += tombstones_dropped
        return outcome is not CancellationOutcome.REJECTED

    def _take_pushed_tensor(
        self, spec: ECMooncakeLoadSpec
    ) -> tuple[torch.Tensor, MemoryAllocation]:
        try:
            allocation = self._reservations.take(spec.transfer_id, spec.mm_hash)
        except RuntimeError:
            self._consumer_worker_metrics["takes_rejected"] += 1
            raise
        self._consumer_worker_metrics["reservations_taken"] += 1
        return allocation.tensor, allocation

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
        extra = []
        submit_error: Exception | None = None
        for addr in shards[1:]:
            try:
                extra.append(
                    self._shard_executor().submit(self._reserve_one, addr, spec)
                )
            except Exception as exc:
                submit_error = exc
                break
        first: dict[str, Any] | None = None
        rest: list[dict[str, Any]] = []
        error = submit_error
        if error is None:
            try:
                first = self._reserve_one(shards[0], spec)
            except Exception as exc:
                error = exc
        for future in extra:
            try:
                rest.append(future.result())
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
        assert first is not None
        return [first, *rest]

    def _refresh_remote_reservations(
        self,
        spec: ECMooncakePushSpec,
        reservations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        for shard in reservations:
            if (
                shard.get("ready", False)
                or shard.get("cached", False)
                or shard.get("cancelled", False)
            ):
                continue
            result = self._control_client.request(
                str(shard.get("addr", spec.consumer_zmq)),
                {
                    "op": "cancel",
                    "transfer_id": spec.transfer_id,
                    "reservation_id": str(shard["reservation_id"]),
                    "abandon": True,
                    "refresh": True,
                },
            )
            if not isinstance(result, dict) or not result.get("cancelled"):
                raise RuntimeError(
                    f"Could not refresh EC reservation for mm_hash={spec.mm_hash}"
                )
        return self._reserve_remote(spec)

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
            self._producer_pushes.reserve(
                spec,
                partial(self._submit_reservation, spec),
            )
        if not isinstance(encoder_cache, dict):
            return
        for mm_hash in dict.fromkeys(spec.mm_hash for spec in metadata.pushes):
            tensor = encoder_cache.get(mm_hash)
            if tensor is not None:
                self._bind_push_source(tensor, mm_hash)

    def _submit_reservation(
        self, spec: ECMooncakePushSpec
    ) -> Future[list[dict[str, Any]]]:
        return self._control_executor.submit(self._reserve_remote, spec)

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
        self._reservations.retire_stale(encoder_cache)

        for spec in metadata.loads:
            if spec.mm_hash in encoder_cache:
                if spec.pushed:
                    # The spec's id is one shard's; cancel by transfer.
                    self._cancel_push(spec.transfer_id, "")
                self._completed_loads.add(spec.mm_hash)
                continue
            if spec.local:
                resident = self._consumer_memory.take_resident(
                    spec.mm_hash, tuple(spec.shape), spec.dtype
                )
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

    def _push_batch(self, pushes: list[ProducerPushRecord]) -> None:
        started_at = time.monotonic()
        with self._push_perf_lock:
            self._queued_transfer_batches -= 1
            self._active_transfer_batches += 1

        queue_waits_ms = []
        for push in pushes:
            assert push.source_at is not None
            queue_waits_ms.append(max(0, started_at - push.source_at) * 1000)
        stage_ms = {
            "queue": sum(queue_waits_ms),
            "reserve": 0.0,
            "cuda": 0.0,
            "register": 0.0,
            "rdma": 0.0,
            "unregister": 0.0,
            "complete": 0.0,
        }
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]] = []
        notifications: list[tuple[ProducerPushRecord, dict[str, Any]]] = []
        failed = False
        failure: Exception | None = None
        try:
            synchronized: set[str] = set()
            for push in pushes:
                stage_started_at = time.monotonic()
                reservations = self._producer_pushes.resolve_reservations(push)
                stale = [
                    index
                    for index, shard in enumerate(reservations)
                    if not shard.get("ready", False)
                    and not shard.get("cancelled", False)
                    and time.monotonic() - float(shard.get("_received_at", started_at))
                    >= _RESERVATION_REFRESH_SECONDS
                ]
                if stale:
                    reservations = self._refresh_remote_reservations(
                        push.spec, reservations
                    )
                    self._producer_pushes.replace_reservations(push, reservations)
                stage_ms["reserve"] += (time.monotonic() - stage_started_at) * 1000
                self._producer_pushes.begin_writing(push)
                for shard in reservations:
                    if shard.get("cached", False) or shard.get("cancelled", False):
                        continue
                    if not shard.get("write", True):
                        continue
                    source = push.source
                    assert source is not None
                    if (
                        source.ready_event is not None
                        and push.spec.transfer_id not in synchronized
                    ):
                        stage_started_at = time.monotonic()
                        source.ready_event.synchronize()
                        stage_ms["cuda"] += (time.monotonic() - stage_started_at) * 1000
                        synchronized.add(push.spec.transfer_id)
                    if int(shard["nbytes"]) != source.tensor.nbytes:
                        raise RuntimeError(
                            "Reserved EC size does not match tensor for "
                            f"mm_hash={push.spec.mm_hash}"
                        )
                    ready.append((push, shard))
                    notifications.append((push, shard))
            if ready:
                # One source per push: a sharded consumer reads the same bytes
                # into each of its ranks, so staging and registration happen
                # once however many destinations there are.
                unique: list[ProducerPushRecord] = []
                source_index: dict[str, int] = {}
                for push, _ in ready:
                    transfer_id = push.spec.transfer_id
                    if transfer_id not in source_index:
                        source_index[transfer_id] = len(unique)
                        unique.append(push)
                tensors = [push.source.tensor for push in unique if push.source]
                lengths = [tensor.nbytes for tensor in tensors]
                stage_started_at = time.monotonic()
                staged = self._producer_memory.stage(tensors)
                registered_sources: list[int] = []
                if staged is not None:
                    sources = staged.tensors
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
                    session_records: dict[str, dict[str, ProducerPushRecord]] = {}
                    for push, shard in ready:
                        session = str(shard["dst_session"])
                        by_session.setdefault(session, []).append(
                            (source_index[push.spec.transfer_id], int(shard["dst_ptr"]))
                        )
                        session_records.setdefault(session, {})[
                            push.spec.transfer_id
                        ] = push
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
                    write_error: Exception | None = None
                    extra = []
                    for session, items in sessions[1:]:
                        try:
                            future = self._shard_executor().submit(
                                write, session, items
                            )
                        except Exception as exc:
                            write_error = exc
                            break
                        extra.append(future)
                        self._producer_pushes.track_shard_futures(
                            list(session_records[session].values()), [future]
                        )
                    if write_error is None:
                        try:
                            write(*sessions[0])
                        except Exception as exc:
                            write_error = exc
                    for future in extra:
                        try:
                            future.result()
                        except Exception as exc:
                            if write_error is None:
                                write_error = exc
                    if write_error is not None:
                        raise write_error
                    stage_ms["rdma"] = (time.monotonic() - stage_started_at) * 1000
                finally:
                    stage_started_at = time.monotonic()
                    if staged is not None:
                        self._producer_memory.release(staged)
                    self._transfer.release_sources(registered_sources)
                    stage_ms["unregister"] = (
                        time.monotonic() - stage_started_at
                    ) * 1000

            self._producer_pushes.begin_notifying(pushes)
            stage_started_at = time.monotonic()
            self._notify_completions(notifications)
            stage_ms["complete"] = (time.monotonic() - stage_started_at) * 1000
            self._producer_pushes.complete(pushes)
        except Exception as exc:
            # A failed batch must not take the engine down with it: the
            # consumer is told to drop its reservations and this item falls
            # back to whatever the consumer can still do (pull, or a local
            # re-encode). Raising here would surface in
            # `build_connector_worker_meta` as a fatal EngineCore error.
            failed = True
            failure = exc
            logger.exception(
                "EC Mooncake push batch failed for mm_hashes=%s",
                [push.spec.mm_hash for push in pushes],
            )
            self._producer_pushes.settle_all(pushes)
            self._abandon_pushes(pushes)
        finally:
            if failure is not None:
                self._producer_pushes.fail(pushes, failure)
            stage_ms["total"] = (time.monotonic() - started_at) * 1000
            self._record_push_perf(
                stage_ms,
                stage_max_ms={"queue": max(queue_waits_ms, default=0.0)},
                item_count=len(pushes),
                # `ready` holds one entry per destination shard, so count the
                # distinct items rather than the writes.
                byte_count=sum(
                    push.spec.nbytes
                    for push in {p.spec.transfer_id: p for p, _ in ready}.values()
                ),
                skipped_items=len(pushes)
                - len({push.spec.transfer_id for push, _ in ready}),
                failed=failed,
            )

    def _notify_completions(
        self, notifications: list[tuple[ProducerPushRecord, dict[str, Any]]]
    ) -> None:
        """Tell the consumer, in one message per destination, what landed."""
        if not notifications:
            return
        by_destination: dict[str, list[tuple[ProducerPushRecord, dict[str, Any]]]] = {}
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

    def _abandon_pushes(self, pushes: list[ProducerPushRecord]) -> None:
        """Release the consumer-side reservations of a batch that failed."""
        for push in pushes:
            shards = push.reservations
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
        self._producer_pushes.submit_batches(
            self._io_executor,
            self._push_batch,
            self._note_push_batch_queued,
        )

    def _note_push_batch_queued(self) -> None:
        with self._push_perf_lock:
            self._queued_transfer_batches += 1

    def _bind_push_source(self, tensor: torch.Tensor, mm_hash: str) -> None:
        ready_event = None
        if tensor.device.type == "cuda":
            ready_event = torch.Event()
            ready_event.record(torch.accelerator.current_stream(tensor.device))
        self._producer_pushes.bind_source(mm_hash, tensor, ready_event)

    def _cancel_orphaned_reservation(self, record: ProducerPushRecord) -> None:
        try:
            for shard in self._producer_pushes.resolve_reservations(record):
                if shard.get("cached", False) or shard.get("cancelled", False):
                    continue
                self._control_client.request(
                    str(shard.get("addr", record.spec.consumer_zmq)),
                    {
                        "op": "cancel",
                        "transfer_id": record.spec.transfer_id,
                        "reservation_id": str(shard.get("reservation_id", "")),
                        "abandon": True,
                    },
                )
        except Exception:
            logger.exception(
                "Failed to cancel orphaned EC reservation for transfer_id=%s",
                record.spec.transfer_id,
            )
        finally:
            self._producer_pushes.finish_cancel(record)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if not self.is_producer:
            return None, None

        for record in self._producer_pushes.cancel_requests(finished_req_ids):
            self._producer_pushes.submit_cancel(
                record,
                self._io_executor,
                self._cancel_orphaned_reservation,
            )
        return None, None

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        if not self.is_producer:
            return
        tensor = encoder_cache[mm_hash]
        self._bind_push_source(tensor, mm_hash)

    def build_connector_worker_meta(self) -> ECMooncakeWorkerMetadata | None:
        if self.is_consumer and not self._is_receiving_rank:
            # `loaded` is intersected across reporting ranks, so a stage that
            # never loads must not report at all rather than report nothing.
            return None

        self._flush_pending_pushes()
        failures = self._producer_pushes.poll()
        self._producer_metrics["saves_failed"] += len(failures)
        for mm_hash, error in failures:
            logger.error(
                "EC Mooncake async save failed for mm_hash=%s: %s",
                mm_hash,
                error,
            )
        reclaimed = self._consumer_memory.drain_reclaimed()
        meta = ECMooncakeWorkerMetadata(
            loaded=self._completed_loads,
            failed_loads=self._failed_loads,
            reclaimed=reclaimed,
            pending_loads=False,
            pending_saves=self._producer_pushes.pending,
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
        self._consumer_memory.close()
        self._producer_memory.close()
        self._transfer.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
