# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Collection
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import zmq

from vllm.config import ModelConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
    ECMooncakePushSpec,
    ECMooncakeWorkerMetadata,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

logger = init_logger(__name__)

_LEASE_TTL_SECONDS = 300
_DRAIN_MIN_INTERVAL = 0.005
_MAX_PENDING_EVENTS = 4096
_MAX_CANCELLED_TRANSFER_IDS = 1 << 16


@dataclass(frozen=True)
class ECMooncakeSchedulerConfig:
    is_producer: bool
    is_consumer: bool
    reservation_zmq_addr: str | None
    consumer_pool_capacity: int
    push_wait_timeout: float
    consumer_metrics_log_interval: float
    encoder_cache_hidden_dim: int | None


class ECMooncakeScheduler:
    def __init__(
        self,
        config: ECMooncakeSchedulerConfig,
        model_config: ModelConfig,
        control_request: Callable[[str, dict[str, Any]], Any],
        submit_control: Callable[..., Future[Any]],
    ) -> None:
        self._is_producer = config.is_producer
        self._is_consumer = config.is_consumer
        self._reservation_zmq_addr = config.reservation_zmq_addr
        self._consumer_pool_capacity = config.consumer_pool_capacity
        self._push_wait_timeout = config.push_wait_timeout
        self._consumer_metrics_log_interval = config.consumer_metrics_log_interval
        self._encoder_cache_hidden_dim = config.encoder_cache_hidden_dim
        self._model_config = model_config
        self._control_request = control_request
        self._submit_control = submit_control

        self._metadata_fields_cache: dict[str, set[str]] = {}
        self._consumer_metrics_started_at = time.monotonic()
        self._consumer_scheduler_metrics: Counter[str] = Counter()
        self._consumer_missing_since: dict[str, float] = {}
        self._stalled_hashes: set[str] = set()
        self._unavailable_requests: set[str] = set()
        self._drain_pending = True
        self._drained_at = 0.0
        self._consumer_loading_since: dict[str, float] = {}
        self._consumer_pending_since: dict[str, float] = {}
        self._pending_spec_deadlines: dict[str, float] = {}
        self._pending_cancels: dict[str, Future[Any]] = {}
        self._cancelled_transfer_ids: OrderedDict[str, float] = OrderedDict()
        self._pending_specs: dict[str, ECMooncakeLoadSpec] = {}
        self._pending_specs_by_hash: dict[str, deque[str]] = {}
        self._load_specs: dict[str, ECMooncakeLoadSpec] = {}
        self._mm_datas_need_loads: dict[str, int] = {}
        self._loading_hashes: set[str] = set()
        self._ready_hashes: set[str] = set()
        self._resident_specs: OrderedDict[str, ECMooncakeLoadSpec] = OrderedDict()
        self._resident_bytes = 0
        self._scheduler_pending_work = False
        self._pushes_to_prepare: dict[str, ECMooncakePushSpec] = {}
        self._prepared_push_transfer_ids: set[str] = set()
        self._consumer_shard_cache: dict[str, list[str]] = {}
        self._event_zmq_ctx: zmq.Context | None = None
        self._event_zmq_socket: zmq.Socket | None = None
        self._event_shard_count = 1
        self._event_ready_shards: OrderedDict[str, set[int]] = OrderedDict()

    def _consumer_shards(self, base_addr: str) -> list[str]:
        cached = self._consumer_shard_cache.get(base_addr)
        if cached is not None:
            return cached
        shards = [base_addr]
        try:
            reply = self._control_request(base_addr, {"op": "peers"})
            ports = reply.get("ports") if isinstance(reply, dict) else None
            if ports:
                prefix = base_addr.rsplit(":", 1)[0]
                shards = [f"{prefix}:{int(port)}" for port in ports]
        except Exception:
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

    def _cancel_remote(
        self, consumer_zmq: str, transfer_id: str, reservation_id: str
    ) -> bool:
        cancelled = False
        for addr in self._consumer_shards(consumer_zmq):
            result = self._control_request(
                addr,
                {
                    "op": "cancel",
                    "transfer_id": transfer_id,
                    "reservation_id": reservation_id,
                },
            )
            cancelled |= isinstance(result, dict) and bool(result.get("cancelled"))
        return cancelled

    def _clear_item_timers(self, mm_hash: str) -> None:
        self._consumer_missing_since.pop(mm_hash, None)
        self._consumer_loading_since.pop(mm_hash, None)
        self._consumer_pending_since.pop(mm_hash, None)
        self._stalled_hashes.discard(mm_hash)

    def _note_awaiting_push(
        self,
        mm_hash: str,
        transfer_id: str | None = None,
        request_id: str | None = None,
    ) -> bool:
        now = time.monotonic()
        since = self._consumer_missing_since.setdefault(mm_hash, now)
        self._consumer_scheduler_metrics["missing_event"] += 1
        elapsed = now - since
        if elapsed < self._push_wait_timeout:
            return False
        stale = mm_hash in self._stalled_hashes
        if request_id is not None:
            self._unavailable_requests.add(request_id)
            self._consumer_scheduler_metrics["given_up"] += 1
            self._consumer_missing_since.pop(mm_hash, None)
        if stale:
            return request_id is not None
        self._stalled_hashes.add(mm_hash)
        self._consumer_scheduler_metrics["stalled"] += 1
        reservation: Any = "unknown"
        if transfer_id and self._reservation_zmq_addr is not None:
            try:
                reservation = self._control_request(
                    self._reservation_zmq_addr,
                    {"op": "status", "transfer_id": transfer_id},
                )
            except Exception as e:  # noqa: BLE001 - diagnostic only
                reservation = f"status failed: {e}"
        logger.warning(
            "EC Mooncake waited %.1fs for a push of mm_hash=%s "
            "(transfer_id=%s) that never arrived; worker reservation=%s; "
            "requests needing it fail with a retryable error.",
            elapsed,
            mm_hash,
            transfer_id,
            reservation,
        )
        return request_id is not None

    def take_unavailable_requests(self) -> set[str]:
        given_up = self._unavailable_requests
        self._unavailable_requests = set()
        return given_up

    @staticmethod
    def _hash_samples(values: list[str], limit: int = 5) -> list[str]:
        return [value[:16] for value in values[:limit]]

    def _maybe_log_consumer_scheduler_metrics(self) -> None:
        now = time.monotonic()
        if (
            self._consumer_metrics_log_interval <= 0
            or now - self._consumer_metrics_started_at
            < self._consumer_metrics_log_interval
        ):
            return
        missing = sorted(self._consumer_missing_since.items(), key=lambda item: item[1])
        loading = sorted(self._consumer_loading_since.items(), key=lambda item: item[1])
        pending = sorted(self._consumer_pending_since.items(), key=lambda item: item[1])
        oldest_missing_ms = round((now - missing[0][1]) * 1000, 1) if missing else 0.0
        oldest_loading_ms = round((now - loading[0][1]) * 1000, 1) if loading else 0.0
        oldest_pending_ms = round((now - pending[0][1]) * 1000, 1) if pending else 0.0
        logger.info(
            "EC Mooncake consumer scheduler: decisions=%s, ready=%d, loading=%d, "
            "resident=%d, pending_specs=%d, needs_load=%d, missing=%d, "
            "oldest_missing_ms=%.1f, oldest_loading_ms=%.1f, "
            "oldest_pending_ms=%.1f, missing_hashes=%s, loading_hashes=%s, "
            "pending_hashes=%s",
            dict(self._consumer_scheduler_metrics),
            len(self._ready_hashes),
            len(self._loading_hashes),
            len(self._resident_specs),
            len(self._pending_specs),
            len(self._mm_datas_need_loads),
            len(missing),
            oldest_missing_ms,
            oldest_loading_ms,
            oldest_pending_ms,
            self._hash_samples([mm_hash for mm_hash, _ in missing]),
            self._hash_samples([mm_hash for mm_hash, _ in loading]),
            self._hash_samples([mm_hash for mm_hash, _ in pending]),
        )
        self._consumer_scheduler_metrics.clear()
        self._consumer_metrics_started_at = now

    @staticmethod
    def _expire_cancel_records(records: OrderedDict[str, float], now: float) -> int:
        dropped = 0
        while records:
            expires_at = next(iter(records.values()))
            if expires_at > now and len(records) <= _MAX_CANCELLED_TRANSFER_IDS:
                break
            records.popitem(last=False)
            dropped += 1
        return dropped

    def _poll_pending_cancels(self) -> None:
        pending = {}
        for transfer_id, future in self._pending_cancels.items():
            if not future.done():
                pending[transfer_id] = future
                continue
            try:
                cancelled = future.result()
            except Exception:
                self._cancelled_transfer_ids.pop(transfer_id, None)
                self._consumer_scheduler_metrics["cancellations_failed"] += 1
                logger.warning(
                    "EC Mooncake reservation cancellation failed", exc_info=True
                )
            else:
                key = "cancellations_completed" if cancelled else "cancellations_stale"
                self._consumer_scheduler_metrics[key] += 1
        self._pending_cancels = pending

    def _index_pending_spec(self, spec: ECMooncakeLoadSpec) -> None:
        transfer_id = spec.transfer_id or spec.mm_hash
        if transfer_id in self._pending_specs:
            self._consumer_scheduler_metrics["events_duplicate"] += 1
            return
        self._pending_specs[transfer_id] = spec
        self._pending_specs_by_hash.setdefault(spec.mm_hash, deque()).append(
            transfer_id
        )
        self._pending_spec_deadlines[transfer_id] = (
            time.monotonic() + _LEASE_TTL_SECONDS
        )
        self._consumer_missing_since.pop(spec.mm_hash, None)
        self._consumer_pending_since.setdefault(spec.mm_hash, time.monotonic())

    def _pop_pending_spec(self, transfer_id: str) -> ECMooncakeLoadSpec | None:
        spec = self._pending_specs.pop(transfer_id, None)
        self._pending_spec_deadlines.pop(transfer_id, None)
        self._forget_shard_readiness(transfer_id)
        if spec is not None:
            if not self._pending_specs_by_hash.get(spec.mm_hash):
                self._consumer_pending_since.pop(spec.mm_hash, None)
            transfer_ids = self._pending_specs_by_hash.get(spec.mm_hash)
            if transfer_ids is not None:
                with suppress(ValueError):
                    transfer_ids.remove(transfer_id)
                if not transfer_ids:
                    self._pending_specs_by_hash.pop(spec.mm_hash, None)
                    self._consumer_pending_since.pop(spec.mm_hash, None)
            else:
                self._consumer_pending_since.pop(spec.mm_hash, None)
        return spec

    def _first_pending_spec(self, mm_hash: str) -> ECMooncakeLoadSpec | None:
        transfer_ids = self._pending_specs_by_hash.get(mm_hash)
        if transfer_ids is None:
            return None
        while transfer_ids:
            spec = self._pending_specs.get(transfer_ids[0])
            if spec is not None:
                return spec
            transfer_ids.popleft()
        self._pending_specs_by_hash.pop(mm_hash, None)
        self._consumer_pending_since.pop(mm_hash, None)
        return None

    def _note_shard_ready(self, data: dict[str, Any]) -> bool:
        if self._event_shard_count <= 1:
            return True
        transfer_id = str(data["transfer_id"])
        if transfer_id in self._pending_specs:
            return False
        shard = data.get("shard")
        shards = self._event_ready_shards.setdefault(transfer_id, set())
        self._event_ready_shards.move_to_end(transfer_id)
        shards.add(int(shard) if shard is not None else len(shards))
        if len(shards) < self._event_shard_count:
            self._consumer_scheduler_metrics["events_awaiting_shards"] += 1
            while len(self._event_ready_shards) > _MAX_PENDING_EVENTS:
                self._event_ready_shards.popitem(last=False)
                self._consumer_scheduler_metrics["events_partial_dropped"] += 1
            return False
        self._event_ready_shards.pop(transfer_id, None)
        self._consumer_scheduler_metrics["events_all_shards_ready"] += 1
        return True

    def _forget_shard_readiness(self, transfer_id: str) -> None:
        self._event_ready_shards.pop(transfer_id, None)

    def _store_pushed_spec(self, data: dict[str, Any]) -> None:
        transfer_id = str(data["transfer_id"])
        identifier = str(data["mm_hash"])
        reservation_id = str(data["reservation_id"])
        self._index_pending_spec(
            ECMooncakeLoadSpec(
                mm_hash=identifier,
                num_token=0,
                nbytes=int(data["nbytes"]),
                shape=tuple(int(value) for value in data["shape"]),
                dtype=str(data["dtype"]),
                pushed=True,
                transfer_id=transfer_id,
                reservation_id=reservation_id,
            )
        )

    def _note_resident(self, spec: ECMooncakeLoadSpec) -> None:
        self._drop_resident(spec.mm_hash)
        self._resident_specs[spec.mm_hash] = ECMooncakeLoadSpec(
            mm_hash=spec.mm_hash,
            num_token=0,
            nbytes=spec.nbytes,
            shape=spec.shape,
            dtype=spec.dtype,
            local=True,
        )
        self._resident_bytes += spec.nbytes
        while (
            self._resident_specs and self._resident_bytes > self._consumer_pool_capacity
        ):
            _, dropped = self._resident_specs.popitem(last=False)
            self._resident_bytes -= dropped.nbytes

    def _drop_resident(self, mm_hash: str) -> None:
        spec = self._resident_specs.pop(mm_hash, None)
        if spec is not None:
            self._resident_bytes -= spec.nbytes

    def _queue_cancel(self, transfer_id: str, reservation_id: str = "") -> None:
        if (
            self._reservation_zmq_addr is None
            or transfer_id in self._pending_cancels
            or transfer_id in self._cancelled_transfer_ids
        ):
            return
        self._cancelled_transfer_ids[transfer_id] = (
            time.monotonic() + _LEASE_TTL_SECONDS
        )
        self._pending_cancels[transfer_id] = self._submit_control(
            self._cancel_remote,
            self._reservation_zmq_addr,
            transfer_id,
            reservation_id,
        )

    def _expire_pending_specs(self) -> None:
        now = time.monotonic()
        for transfer_id, deadline in list(self._pending_spec_deadlines.items()):
            if deadline > now:
                continue
            spec = self._pop_pending_spec(transfer_id)
            if spec is not None:
                self._consumer_pending_since.pop(spec.mm_hash, None)
                self._consumer_scheduler_metrics["pending_specs_expired"] += 1
                self._queue_cancel(transfer_id)

    def _ensure_event_channel(self) -> None:
        if self._event_zmq_socket is not None:
            return
        assert self._reservation_zmq_addr is not None
        shards = self._consumer_shards(self._reservation_zmq_addr)
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PULL)
        connected = 0
        for addr in shards:
            try:
                event_port = self._control_request(addr, {"op": "event_port"})
                address, _ = addr.rsplit(":", 1)
                socket.connect(f"{address}:{int(event_port)}")
            except Exception:
                logger.warning(
                    "EC Mooncake could not subscribe to the event channel of "
                    "consumer shard %s; its readiness will only be seen "
                    "through reserve replies.",
                    addr,
                )
                continue
            connected += 1
        if not connected:
            socket.close(linger=0)
            ctx.term()
            return
        self._event_zmq_ctx = ctx
        self._event_zmq_socket = socket
        self._event_shard_count = connected

    def _drain_push_notifications(self) -> None:
        now = time.monotonic()
        if not self._drain_pending and now - self._drained_at < _DRAIN_MIN_INTERVAL:
            return
        self._drain_pending = False
        self._drained_at = now
        self._poll_pending_cancels()
        self._expire_pending_specs()
        self._consumer_scheduler_metrics["cancel_records_dropped"] += (
            self._expire_cancel_records(self._cancelled_transfer_ids, now)
        )
        if self._reservation_zmq_addr is not None:
            self._ensure_event_channel()
        socket = self._event_zmq_socket
        if socket is None:
            return
        while True:
            try:
                data = socket.recv_json(flags=zmq.DONTWAIT)
            except zmq.Again:
                return
            identifier = str(data["mm_hash"])
            self._consumer_scheduler_metrics["events_received"] += 1
            if data.get("ready"):
                self._consumer_scheduler_metrics["events_ready"] += 1
                transfer_id = str(data["transfer_id"])
                if transfer_id in self._cancelled_transfer_ids:
                    self._consumer_scheduler_metrics["events_cancelled"] += 1
                    continue
                if identifier in self._ready_hashes:
                    self._consumer_scheduler_metrics["events_redundant"] += 1
                if not self._note_shard_ready(data):
                    continue
                self._store_pushed_spec(data)
            else:
                self._consumer_scheduler_metrics["events_not_ready"] += 1

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        self._drain_push_notifications()
        self._maybe_log_consumer_scheduler_metrics()
        if identifier in self._ready_hashes:
            self._consumer_scheduler_metrics["ready"] += 1
            self._clear_item_timers(identifier)
            return True
        if identifier in self._loading_hashes:
            self._consumer_scheduler_metrics["loading"] += 1
            return False
        if identifier in self._resident_specs:
            self._consumer_scheduler_metrics["resident"] += 1
            self._consumer_missing_since.pop(identifier, None)
            return True
        pending = self._first_pending_spec(identifier)
        if pending is not None:
            self._consumer_scheduler_metrics["pending_spec"] += 1
            self._consumer_missing_since.pop(identifier, None)
            return True
        self._consumer_scheduler_metrics["missing_event"] += 1
        self._consumer_missing_since.setdefault(identifier, time.monotonic())
        return False

    @staticmethod
    def _request_transfer_id(request: Any, index: int) -> str | None:
        params = getattr(request, "ec_transfer_params", None) or {}
        items = params.get("ec_items") or []
        mm_hash = request.mm_features[index].identifier
        if index < len(items):
            item = items[index]
            if item.get("mm_hash") in (None, mm_hash) and item.get("transfer_id"):
                return str(item["transfer_id"])
        for item in items:
            if item.get("mm_hash") == mm_hash and item.get("transfer_id"):
                return str(item["transfer_id"])
        return None

    def ensure_cache_available(
        self,
        request: Any,
        num_computed_tokens: int,
        local_cache_hashes: Collection[str] | None = None,
    ) -> bool:
        if self._is_producer:
            for index, feature in enumerate(request.mm_features):
                if (
                    feature.mm_position.offset + feature.mm_position.length
                    > num_computed_tokens
                ):
                    self._prepare_push_spec(request, index)
        if not self._is_consumer:
            return True

        self._drain_push_notifications()
        local_cache_hashes = local_cache_hashes or set()
        all_ready = True
        for index, feature in enumerate(request.mm_features):
            if (
                feature.mm_position.offset + feature.mm_position.length
                <= num_computed_tokens
            ):
                continue
            mm_hash = feature.identifier
            transfer_id = self._request_transfer_id(request, index)
            if transfer_id is not None and transfer_id in self._pending_spec_deadlines:
                self._pending_spec_deadlines[transfer_id] = (
                    time.monotonic() + _LEASE_TTL_SECONDS
                )
            if mm_hash in local_cache_hashes:
                continue
            if mm_hash in self._ready_hashes:
                self._consumer_scheduler_metrics["ready"] += 1
                self._clear_item_timers(mm_hash)
                continue
            if mm_hash in self._loading_hashes:
                self._consumer_scheduler_metrics["loading"] += 1
                all_ready = False
                continue
            spec = self._resident_specs.get(mm_hash)
            if spec is not None:
                self._consumer_scheduler_metrics["resident_hit"] += 1
            else:
                spec = (
                    self._pending_specs.get(transfer_id)
                    if transfer_id is not None
                    else None
                )
                if spec is None:
                    spec = self._first_pending_spec(mm_hash)
            if spec is not None:
                self._loading_hashes.add(mm_hash)
                self._load_specs[mm_hash] = spec
                self._consumer_loading_since.setdefault(mm_hash, time.monotonic())
                self._consumer_pending_since.pop(mm_hash, None)
                self._mm_datas_need_loads[mm_hash] = request.get_num_encoder_embeds(
                    index
                )
                self._scheduler_pending_work = True
                all_ready = False
            else:
                self._note_awaiting_push(mm_hash, transfer_id, request.request_id)
                all_ready = False
        return all_ready

    def _prepare_push_spec(self, request: Any, index: int) -> None:
        params = getattr(request, "ec_transfer_params", None) or {}
        consumer_zmq = params.get("consumer_zmq")
        mm_hash = request.mm_features[index].identifier
        transfer_id = self._request_transfer_id(request, index)
        if transfer_id is None:
            transfer_id = f"{request.request_id}:{index}"
        if not consumer_zmq or transfer_id in self._prepared_push_transfer_ids:
            return
        num_tokens = request.get_num_encoder_embeds(index)
        dtype = self._model_config.dtype
        assert isinstance(dtype, torch.dtype)
        assert self._encoder_cache_hidden_dim is not None
        dtype_name = str(dtype).split(".")[-1]
        shape = (num_tokens, self._encoder_cache_hidden_dim)
        nbytes = math.prod(shape) * dtype.itemsize
        self._pushes_to_prepare[transfer_id] = ECMooncakePushSpec(
            mm_hash=mm_hash,
            nbytes=nbytes,
            shape=shape,
            dtype=dtype_name,
            consumer_zmq=str(consumer_zmq),
            transfer_id=transfer_id,
            request_id=request.request_id,
        )
        self._prepared_push_transfer_ids.add(transfer_id)

    def update_state_after_alloc(self, request: Any, index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        if self._is_producer:
            self._prepare_push_spec(request, index)
        if not self._is_consumer:
            return
        if mm_hash in self._ready_hashes:
            return
        if mm_hash in self._loading_hashes:
            return
        num_encoder_token = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def update_state_after_free(self, request: Any, index: int) -> None:
        if not self._is_consumer:
            return
        transfer_id = self._request_transfer_id(request, index)
        if transfer_id is None:
            return
        self._pop_pending_spec(transfer_id)
        self._queue_cancel(transfer_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self._ready_hashes.discard(mm_hash)
            self._clear_item_timers(mm_hash)
        meta = ECMooncakeConnectorMetadata()
        for push_spec in self._pushes_to_prepare.values():
            meta.add_push(push_spec)
        self._pushes_to_prepare.clear()
        for mm_hash, num_token in self._mm_datas_need_loads.items():
            load_spec = self._load_specs.pop(mm_hash, None)
            if load_spec is None:
                logger.warning("Missing EC Mooncake spec for mm_hash=%s", mm_hash)
                continue
            meta.add_load(
                ECMooncakeLoadSpec(
                    mm_hash=load_spec.mm_hash,
                    num_token=num_token,
                    nbytes=load_spec.nbytes,
                    shape=load_spec.shape,
                    dtype=load_spec.dtype,
                    pushed=load_spec.pushed,
                    transfer_id=load_spec.transfer_id,
                    reservation_id=load_spec.reservation_id,
                    local=load_spec.local,
                )
            )
            self._note_resident(load_spec)
            if not load_spec.local:
                self._pop_pending_spec(load_spec.transfer_id or load_spec.mm_hash)
        self._mm_datas_need_loads.clear()
        self._poll_pending_cancels()
        self._maybe_log_consumer_scheduler_metrics()
        self._drain_pending = True
        return meta

    def update_connector_output(self, connector_output: ECConnectorOutput) -> None:
        meta = connector_output.ec_connector_worker_meta
        if not isinstance(meta, ECMooncakeWorkerMetadata):
            return
        for mm_hash in meta.loaded:
            self._loading_hashes.discard(mm_hash)
            self._ready_hashes.add(mm_hash)
            self._clear_item_timers(mm_hash)
            self._consumer_scheduler_metrics["loads_completed"] += 1
        for mm_hash in meta.failed_loads:
            self._loading_hashes.discard(mm_hash)
            self._load_specs.pop(mm_hash, None)
            self._drop_resident(mm_hash)
            self._clear_item_timers(mm_hash)
            self._consumer_scheduler_metrics["loads_failed"] += 1
        for mm_hash in meta.reclaimed:
            self._drop_resident(mm_hash)
            self._consumer_scheduler_metrics["resident_reclaimed"] += 1
        self._scheduler_pending_work = meta.pending_loads or meta.pending_saves

    def has_pending_push_work(self) -> bool:
        return self._scheduler_pending_work

    def _placeholder_metadata_fields(self, modality: str) -> set[str]:
        if modality in self._metadata_fields_cache:
            return self._metadata_fields_cache[modality]

        fields: set[str] = set()
        try:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            info = MULTIMODAL_REGISTRY.create_processor(self._model_config).info
            fields = info.data_parser.placeholder_metadata_fields(modality)
        except Exception:
            logger.warning(
                "Could not determine the placeholder metadata fields for "
                "modality %s; the consumer will preprocess the media itself.",
                modality,
                exc_info=True,
            )

        self._metadata_fields_cache[modality] = fields
        return fields

    def request_finished(self, request: Any) -> tuple[bool, dict[str, Any] | None]:
        if self._is_consumer:
            for index in range(len(request.mm_features)):
                transfer_id = self._request_transfer_id(request, index)
                if transfer_id is None:
                    continue
                self._pop_pending_spec(transfer_id)
                self._queue_cancel(transfer_id)
        if self._is_producer and self._prepared_push_transfer_ids:
            for index in range(len(request.mm_features)):
                transfer_id = self._request_transfer_id(request, index)
                if transfer_id is None:
                    transfer_id = f"{request.request_id}:{index}"
                self._prepared_push_transfer_ids.discard(transfer_id)
        if not self._is_producer:
            return False, None

        items = []
        for index, feature in enumerate(request.mm_features):
            metadata = {}
            if feature.data is not None:
                wanted = self._placeholder_metadata_fields(feature.modality)
                metadata = {
                    key: value.tolist()
                    for key, value in feature.data.get_data().items()
                    if key in wanted and isinstance(value, torch.Tensor)
                }
            transfer_id = self._request_transfer_id(request, index)
            item = {"mm_hash": feature.identifier, **metadata}
            if transfer_id is not None:
                item["transfer_id"] = transfer_id
            items.append(item)

        if not items:
            return False, None
        return False, {"ec_items": items}

    def close(self) -> None:
        if self._event_zmq_socket is not None:
            self._event_zmq_socket.close(linger=0)
        if self._event_zmq_ctx is not None:
            self._event_zmq_ctx.term()
