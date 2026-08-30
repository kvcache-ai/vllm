# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata exchanged by the Mooncake encoder-cache connector."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorWorkerMetadata,
)


@dataclass
class ECMooncakeLoadSpec:
    """Per-item metadata shipped from scheduler to worker (pickle-friendly)."""

    mm_hash: str
    num_token: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    pushed: bool = False
    transfer_id: str = ""
    reservation_id: str = ""
    # The consumer pool still holds this item, so the load is a local handoff:
    # no transfer, no producer.
    local: bool = False


@dataclass
class ECMooncakePushSpec:
    """Destination reservation requested before an encoder tensor is ready."""

    mm_hash: str
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    consumer_zmq: str
    transfer_id: str
    request_id: str = ""


@dataclass
class ECMooncakeConnectorMetadata(ECConnectorMetadata):
    """Worker-side metadata for one scheduler step."""

    loads: list[ECMooncakeLoadSpec] = field(default_factory=list)
    pushes: list[ECMooncakePushSpec] = field(default_factory=list)

    def add_load(self, spec: ECMooncakeLoadSpec) -> None:
        self.loads.append(spec)

    def add_push(self, spec: ECMooncakePushSpec) -> None:
        self.pushes.append(spec)


@dataclass
class ECMooncakeWorkerMetadata(ECConnectorWorkerMetadata):
    """Completion state reported from workers to the scheduler."""

    loaded: set[str] = field(default_factory=set)
    failed_loads: set[str] = field(default_factory=set)
    # Items the receive pool dropped under pressure. The scheduler assumes an
    # evicted item stays resident until told otherwise.
    reclaimed: set[str] = field(default_factory=set)
    pending_loads: bool = False
    pending_saves: bool = False

    def aggregate(self, other: ECConnectorWorkerMetadata) -> ECMooncakeWorkerMetadata:
        assert isinstance(other, ECMooncakeWorkerMetadata)
        return ECMooncakeWorkerMetadata(
            # Every tensor-parallel rank gathers the embedding from its own
            # cache, so an item counts as loaded only where all of them have
            # it; one rank falling short must fail the load rather than leave
            # the scheduler believing it is ready.
            loaded=self.loaded & other.loaded,
            failed_loads=self.failed_loads | other.failed_loads,
            reclaimed=self.reclaimed | other.reclaimed,
            pending_loads=self.pending_loads or other.pending_loads,
            pending_saves=self.pending_saves or other.pending_saves,
        )
