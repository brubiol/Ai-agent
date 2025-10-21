from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.task_manager import TaskManager


class DummyRequest:
    def __init__(self) -> None:
        self._disconnected = asyncio.Event()

    async def is_disconnected(self) -> bool:
        if self._disconnected.is_set():
            return True
        await asyncio.sleep(0)
        return False

    def trigger_disconnect(self) -> None:
        self._disconnected.set()


pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_tasks_cancelled_when_disconnect_detected(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    manager = TaskManager()
    request_id = "test-request"
    request = DummyRequest()

    cancelled = asyncio.Event()

    async def long_running() -> None:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    worker = asyncio.create_task(long_running())
    await manager.register_task(request_id, worker)

    watcher = await manager.create_task(
        request_id,
        manager.watch_disconnect(request_id, request, poll_interval=0.01),
    )

    request.trigger_disconnect()

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert worker.cancelled()

    # ensure a log entry recorded the cancellation for observability
    assert any("Cancelling" in record.message for record in caplog.records)

    watcher.cancel()
    with suppress(asyncio.CancelledError):
        await watcher
    await manager.clear(request_id)
