# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-cache connector backed by Mooncake TransferEngine."""

from __future__ import annotations

from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

import torch

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
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    ECMooncakeWorker,
    _ControlChannel,
    _ensure_mooncake_available,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

__all__ = [
    "ECMooncakeConnector",
    "ECMooncakeConnectorMetadata",
    "ECMooncakeLoadSpec",
    "ECMooncakePushSpec",
    "ECMooncakeWorkerMetadata",
]


class ECMooncakeConnector(ECConnectorBase):
    """Route EC connector operations to the active Mooncake process role."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        _ensure_mooncake_available()

        parallel_config = vllm_config.parallel_config
        ec_cfg = vllm_config.ec_transfer_config
        assert ec_cfg is not None
        if ec_cfg.is_ec_producer:
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

        registered_capacity = int(ec_cfg.ec_buffer_size)
        if registered_capacity <= 0:
            raise ValueError("ECMooncakeConnector requires ec_buffer_size > 0.")

        self._scheduler: ECMooncakeScheduler | None = None
        self._worker: ECMooncakeWorker | None = None
        self._closed = False

        if role == ECConnectorRole.SCHEDULER:
            extra = ec_cfg.ec_connector_extra_config
            control_port_offset = (
                parallel_config.data_parallel_index
                * parallel_config.tensor_parallel_size
            )
            reservation_port = extra.get("reservation_zmq_port")
            reservation_zmq_addr: str | None = extra.get("reservation_zmq_addr")
            if reservation_zmq_addr is None and reservation_port is not None:
                base = int(reservation_port) + control_port_offset
                reservation_zmq_addr = f"tcp://127.0.0.1:{base}"
            if self.is_consumer and not reservation_zmq_addr:
                raise ValueError(
                    "ec_consumer with ECMooncakeConnector requires "
                    "reservation_zmq_port or reservation_zmq_addr."
                )

            consumer_pool_capacity = int(
                extra.get("consumer_buffer_pool_size", registered_capacity)
            )
            consumer_metrics_log_interval = float(
                extra.get("consumer_metrics_log_interval", 10)
            )
            control_channel = _ControlChannel(
                int(float(extra.get("control_timeout_s", 30)) * 1000)
            )
            control_executor = ThreadPoolExecutor(
                max_workers=int(extra.get("control_max_workers", 8)),
                thread_name_prefix="ec-mooncake-control",
            )

            def close_control() -> None:
                control_executor.shutdown(wait=True, cancel_futures=True)
                control_channel.close()

            encoder_cache_hidden_dim = (
                _get_encoder_cache_hidden_dim(vllm_config) if self.is_producer else None
            )
            self._scheduler = ECMooncakeScheduler(
                ECMooncakeSchedulerConfig(
                    is_producer=self.is_producer,
                    is_consumer=self.is_consumer,
                    reservation_zmq_addr=reservation_zmq_addr,
                    consumer_pool_capacity=consumer_pool_capacity,
                    push_wait_timeout=float(extra.get("push_wait_timeout_s", 60)),
                    consumer_metrics_log_interval=consumer_metrics_log_interval,
                    encoder_cache_hidden_dim=encoder_cache_hidden_dim,
                ),
                model_config=vllm_config.model_config,
                control_request=control_channel.request,
                submit_control=control_executor.submit,
                close_control=close_control,
            )
        else:
            self._worker = ECMooncakeWorker(vllm_config)

    def start_worker_services(self) -> None:
        assert self._worker is not None
        self._worker.start_services()

    def start_save_caches(self, **kwargs: Any) -> None:
        assert self._worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        self._worker.start_save_caches(metadata, **kwargs)

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        assert self._worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMooncakeConnectorMetadata)
        self._worker.start_load_caches(metadata, encoder_cache, **kwargs)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        assert self._worker is not None
        self._worker.save_caches(encoder_cache, mm_hash, **kwargs)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        assert self._worker is not None
        return self._worker.get_finished(finished_req_ids)

    def build_connector_worker_meta(self) -> ECConnectorWorkerMetadata | None:
        assert self._worker is not None
        return self._worker.build_connector_worker_meta()

    def take_unavailable_requests(self) -> set[str]:
        assert self._scheduler is not None
        return self._scheduler.take_unavailable_requests()

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
        if self._closed:
            return
        self._closed = True
        if self._scheduler is not None:
            self._scheduler.close()
        if self._worker is not None:
            self._worker.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.shutdown()
