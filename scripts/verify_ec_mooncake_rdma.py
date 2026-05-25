#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""RDMA/TCP EC transfer with explicit checksum report for manual verification."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import socket
import time
from unittest.mock import Mock

import httpx
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
    ECMooncakeConnectorMetadata,
    ECMooncakeLoadSpec,
)

MM_HASH = "verify_mm_hash_rdma"
SEED = 12345
SHAPE = (8, 64)


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _protocol() -> str:
    return os.environ.get("MOONCAKE_EC_PROTOCOL", "rdma")


def _mock_producer(port: int) -> Mock:
    cfg = Mock(spec=VllmConfig)
    cfg.parallel_config = Mock(tensor_parallel_size=1, pipeline_parallel_size=1)
    cfg.ec_transfer_config = Mock(
        is_ec_producer=True,
        is_ec_consumer=False,
        ec_buffer_device="cuda",
        ec_connector_extra_config={
            "mooncake_protocol": _protocol(),
            "registry_http_port": port,
        },
    )
    return cfg


def _mock_consumer() -> Mock:
    cfg = Mock(spec=VllmConfig)
    cfg.parallel_config = Mock(tensor_parallel_size=1, pipeline_parallel_size=1)
    cfg.ec_transfer_config = Mock(
        is_ec_producer=False,
        is_ec_consumer=True,
        ec_buffer_device="cuda",
        ec_connector_extra_config={
            "mooncake_protocol": _protocol(),
            "remote_registry_url": "http://unused",
        },
    )
    return cfg


def _tensor_meta(t: torch.Tensor) -> dict:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest(),
        "sum": float(t.sum().item()),
        "first4": t.flatten()[:4].tolist(),
    }


def _producer_entry(
    port: int, ready: mp.Queue, done: mp.Event, barrier: mp.Barrier
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.init()
    conn = ECMooncakeConnector(_mock_producer(port), ECConnectorRole.WORKER)
    torch.manual_seed(SEED)
    tensor = torch.randn(*SHAPE, device="cuda", dtype=torch.float32)
    conn.save_caches({MM_HASH: tensor}, MM_HASH)
    ready.put(_tensor_meta(tensor))
    barrier.wait(timeout=120)
    done.wait(timeout=180)


def _consumer_entry(
    registry_url: str, barrier: mp.Barrier, result: mp.Queue
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.init()
    barrier.wait(timeout=120)
    url = f"{registry_url.rstrip('/')}/ec/info/{MM_HASH}"
    r = None
    for _ in range(60):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    if r is None or r.status_code != 200:
        result.put({"ok": False, "err": "registry not ready"})
        return

    reg = r.json()
    spec = ECMooncakeLoadSpec(
        mm_hash=MM_HASH,
        num_token=1,
        nbytes=int(reg["nbytes"]),
        shape=tuple(int(x) for x in reg["shape"]),
        dtype=str(reg["dtype"]),
        producer_zmq=str(reg["producer_zmq"]),
    )
    meta = ECMooncakeConnectorMetadata()
    meta.add_load(spec)
    conn = ECMooncakeConnector(_mock_consumer(), ECConnectorRole.WORKER)
    conn.bind_connector_metadata(meta)
    enc: dict[str, torch.Tensor] = {}
    conn.start_load_caches(enc)
    got = enc.get(MM_HASH)
    if got is None:
        result.put({"ok": False, "err": "missing tensor"})
        return

    torch.manual_seed(SEED)
    expected = torch.randn(*SHAPE, device="cuda", dtype=torch.float32)
    diff = (got.cpu() - expected.cpu()).abs()
    result.put(
        {
            "ok": True,
            "protocol": _protocol(),
            "registry": reg,
            "max_diff": float(diff.max().item()),
            "mean_diff": float(diff.mean().item()),
            "allclose_1e4": bool(
                torch.allclose(got.cpu(), expected.cpu(), atol=1e-4, rtol=0)
            ),
            "expected": _tensor_meta(expected),
            "received": _tensor_meta(got),
        }
    )


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need >= 2 CUDA devices")

    registry_port = _find_free_port()
    registry_url = f"http://127.0.0.1:{registry_port}"
    ctx = mp.get_context("spawn")
    ready: mp.Queue = ctx.Queue()
    result: mp.Queue = ctx.Queue()
    done = ctx.Event()
    barrier = ctx.Barrier(2)

    prod = ctx.Process(
        target=_producer_entry,
        args=(registry_port, ready, done, barrier),
        daemon=True,
    )
    cons = ctx.Process(
        target=_consumer_entry,
        args=(registry_url, barrier, result),
        daemon=True,
    )
    prod.start()
    prod_meta = ready.get(timeout=120)
    cons.start()
    cons.join(timeout=180)
    done.set()
    prod.join(timeout=30)
    out = result.get(timeout=5)

    if not out.get("ok"):
        raise SystemExit(f"FAILED: {out.get('err', out)}")

    exp_sha = out["expected"]["sha256"]
    got_sha = out["received"]["sha256"]

    print("=== Mooncake EC 传输结果校验 ===")
    print(f"protocol:     {out['protocol']}")
    print(f"mm_hash:      {MM_HASH}")
    print(f"registry:     {registry_url}")
    print(f"seed:         {SEED}")
    print(f"tensor shape: {SHAPE}")
    print()
    print("--- Producer (GPU0) 发送前 ---")
    for k, v in prod_meta.items():
        print(f"  {k}: {v}")
    print()
    print("--- Registry /ec/info ---")
    for k, v in out["registry"].items():
        print(f"  {k}: {v}")
    print()
    print("--- Consumer (GPU1) 收到 vs 本地重算期望 ---")
    print(f"  max_diff:       {out['max_diff']:.3e}  (e2e 阈值 < 1e-4)")
    print(f"  mean_diff:      {out['mean_diff']:.3e}")
    print(f"  allclose_1e-4:  {out['allclose_1e4']}")
    print(f"  expected_sha256: {exp_sha}")
    print(f"  received_sha256: {got_sha}")
    print(f"  sha256_match:    {exp_sha == got_sha}")
    print(f"  expected_sum:    {out['expected']['sum']:.6f}")
    print(f"  received_sum:    {out['received']['sum']:.6f}")
    print(f"  expected_first4: {out['expected']['first4']}")
    print(f"  received_first4: {out['received']['first4']}")

    assert out["max_diff"] < 1e-4
    assert exp_sha == got_sha
    print()
    print(">>> 校验结论: PASS（传输后字节级与 seed=12345 期望一致）")


if __name__ == "__main__":
    main()
