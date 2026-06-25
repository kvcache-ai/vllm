# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SGLang-aligned dead-session tracking in MooncakeConnectorWorker.

The worker fast-fails subsequent transfers to a remote session once any
``batch_transfer_sync_write`` to that session returns a non-zero status, so a
single dead RDMA endpoint cannot stall the producer's thread pool.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (  # noqa: E501
    MooncakeConnectorWorker,
)


def _make_worker_stub(engine_returns):
    """Build a minimal object that ``_send_blocks`` can run against."""
    engine = MagicMock()
    engine.batch_transfer_sync_write = MagicMock(side_effect=engine_returns)
    return SimpleNamespace(
        engine=engine,
        _session_lock=threading.Lock(),
        _failed_sessions=set(),
        xfer_stats=MagicMock(),
    )


def _call(worker, session):
    return MooncakeConnectorWorker._send_blocks(
        worker, session, [0x1000], [0x2000], [4096]
    )


def test_failed_session_is_marked_after_engine_returns_nonzero():
    worker = _make_worker_stub(engine_returns=[123])

    assert _call(worker, "host:1") == 123
    assert "host:1" in worker._failed_sessions
    worker.xfer_stats.record_failed_transfer.assert_called_once()


def test_subsequent_transfer_to_dead_session_fast_fails():
    worker = _make_worker_stub(engine_returns=[7])

    _call(worker, "host:1")
    worker.engine.batch_transfer_sync_write.reset_mock()
    worker.xfer_stats.reset_mock()

    assert _call(worker, "host:1") == -1
    worker.engine.batch_transfer_sync_write.assert_not_called()
    worker.xfer_stats.record_failed_transfer.assert_called_once()


def test_other_sessions_unaffected_by_one_failure():
    worker = _make_worker_stub(engine_returns=[5, 0])

    _call(worker, "host:bad")
    assert "host:bad" in worker._failed_sessions

    worker.xfer_stats.reset_mock()
    assert _call(worker, "host:good") == 0
    assert "host:good" not in worker._failed_sessions
    worker.xfer_stats.record_transfer.assert_called_once()


def test_successful_transfer_does_not_mark_session():
    worker = _make_worker_stub(engine_returns=[0])

    assert _call(worker, "host:1") == 0
    assert worker._failed_sessions == set()
    worker.xfer_stats.record_transfer.assert_called_once()
    worker.xfer_stats.record_failed_transfer.assert_not_called()
