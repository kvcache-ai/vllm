# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Control-plane components for the Mooncake encoder-cache connector."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import torch
import zmq
from typing_extensions import NotRequired

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_PENDING_EVENTS = 4096
_RESERVATION_REAP_INTERVAL_SECONDS = 1


class ReservationItem(TypedDict):
    transfer_id: str
    reservation_id: str


class PeersRequest(TypedDict):
    op: Literal["peers"]


class EventPortRequest(TypedDict):
    op: Literal["event_port"]


class StatusRequest(TypedDict):
    op: Literal["status"]
    transfer_id: str


class ReserveRequest(TypedDict):
    op: Literal["reserve"]
    transfer_id: str
    mm_hash: str
    nbytes: int
    shape: list[int]
    dtype: str


class CompleteBatchRequest(TypedDict):
    op: Literal["complete_batch"]
    items: list[ReservationItem]


class ReservationActionRequest(ReservationItem):
    op: Literal["complete", "cancel"]
    abandon: NotRequired[bool]
    refresh: NotRequired[bool]


ControlRequest = (
    PeersRequest
    | EventPortRequest
    | StatusRequest
    | ReserveRequest
    | ReservationActionRequest
    | CompleteBatchRequest
)


class ControlSuccess(TypedDict):
    ok: Literal[True]
    result: NotRequired[Any]


class ControlFailure(TypedDict):
    ok: Literal[False]
    error: str


ControlResponse = ControlSuccess | ControlFailure


@dataclass(frozen=True)
class ControlCompletion:
    accepted: bool
    became_ready: bool = False


class ControlClient:
    """Reusable per-thread REQ sockets for control requests."""

    def __init__(self, timeout_ms: int) -> None:
        self._context = zmq.Context()
        self._timeout_ms = timeout_ms
        self._local = threading.local()
        self._closed = False

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

    def _exchange(self, addr: str, payload: ControlRequest) -> ControlResponse:
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
            self._discard(addr)
            raise
        assert isinstance(response, dict)
        return cast(ControlResponse, response)

    def request(self, addr: str, payload: ControlRequest) -> Any:
        response = self._exchange(addr, payload)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "EC control request failed"))
        return response.get("result")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context.destroy(linger=0)


def make_cancel_request(
    transfer_id: str,
    reservation_id: str,
    *,
    abandon: bool = False,
    refresh: bool = False,
) -> ReservationActionRequest:
    request: ReservationActionRequest = {
        "op": "cancel",
        "transfer_id": transfer_id,
        "reservation_id": reservation_id,
    }
    if abandon:
        request["abandon"] = True
    if refresh:
        request["refresh"] = True
    return request


class ShardTopology:
    """Discover and cache every control address for a consumer."""

    def __init__(self, client: ControlClient) -> None:
        self._client = client
        self._cache: dict[str, list[str]] = {}

    def shards(self, base_addr: str) -> list[str]:
        cached = self._cache.get(base_addr)
        if cached is not None:
            return cached
        shards = [base_addr]
        try:
            reply = self._client.request(base_addr, {"op": "peers"})
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
        self._cache[base_addr] = shards
        if len(shards) > 1:
            logger.info(
                "EC Mooncake consumer at %s has %d shards", base_addr, len(shards)
            )
        return shards


class EventInbox:
    """Non-blocking subscriber for consumer readiness events."""

    def __init__(self, client: ControlClient, topology: ShardTopology) -> None:
        self._client = client
        self._topology = topology
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._closed = False
        self.shard_count = 1

    def _connect(self, base_addr: str) -> None:
        if self._socket is not None:
            return
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        connected = 0
        for addr in self._topology.shards(base_addr):
            try:
                event_port = self._client.request(addr, {"op": "event_port"})
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
            context.term()
            return
        self._context = context
        self._socket = socket
        self.shard_count = connected

    def drain(self, base_addr: str) -> list[dict[str, Any]]:
        self._connect(base_addr)
        if self._socket is None:
            return []
        events = []
        while True:
            try:
                events.append(self._socket.recv_json(flags=zmq.DONTWAIT))
            except zmq.Again:
                return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.close(linger=0)
        if self._context is not None:
            self._context.term()


class ConsumerControlServer:
    """Expose consumer reservations over a ZMQ control channel."""

    def __init__(
        self,
        host: str,
        port: int,
        reserve: Callable[[dict[str, Any]], dict[str, Any]],
        status: Callable[[str], dict[str, Any] | None],
        complete: Callable[[str, str], ControlCompletion],
        cancel: Callable[[str, str, bool, bool], bool],
        reap: Callable[[], int],
        metrics_log_interval: float = 10,
        peer_ports: list[int] | None = None,
        device: torch.device | None = None,
    ) -> None:
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
                torch.accelerator.set_device_index(self._device.index or 0)
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            event_socket = context.socket(zmq.PUSH)
            pending_events: deque[dict[str, Any]] = deque()
            metrics: Counter[str] = Counter()

            def queue_event(event: dict[str, Any]) -> None:
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
                                    transfer_id, str(item["reservation_id"])
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
                                    bool(request.get("refresh", False)),
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

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
