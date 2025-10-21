from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncIterator, Coroutine, Dict, Optional, Set

from fastapi import Request

logger = logging.getLogger(__name__)

TaskType = asyncio.Task[Any]


class TaskManager:
    """Track asyncio tasks spawned per request so they can be canceled together."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Set[TaskType]] = {}
        self._lock = asyncio.Lock()

    async def register_task(self, request_id: str, task: TaskType) -> None:
        async with self._lock:
            tasks = self._tasks.setdefault(request_id, set())
            tasks.add(task)
        task.add_done_callback(lambda finished: asyncio.create_task(self._discard(request_id, finished)))

    async def create_task(self, request_id: str, coro: Coroutine[Any, Any, Any]) -> TaskType:
        task = asyncio.create_task(coro)
        await self.register_task(request_id, task)
        return task

    async def cancel_all(self, request_id: str, *, exclude: Optional[Set[TaskType]] = None) -> None:
        exclude = exclude or set()
        async with self._lock:
            tasks = list(self._tasks.get(request_id, set()))
        to_cancel = [task for task in tasks if task not in exclude and not task.done()]
        if not to_cancel:
            return

        logger.info("Cancelling %d task(s) for request %s", len(to_cancel), request_id)
        for task in to_cancel:
            task.cancel()
        await asyncio.gather(*to_cancel, return_exceptions=True)

    async def clear(self, request_id: str) -> None:
        async with self._lock:
            self._tasks.pop(request_id, None)

    async def watch_disconnect(
        self,
        request_id: str,
        request: Request,
        *,
        poll_interval: float = 0.1,
    ) -> None:
        """Continuously poll for client disconnect and cancel the request's tasks."""
        current = asyncio.current_task()
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnect detected for request %s", request_id)
                    await self.cancel_all(request_id, exclude={current} if current else None)
                    break
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            raise

    async def _discard(self, request_id: str, task: TaskType) -> None:
        async with self._lock:
            tasks = self._tasks.get(request_id)
            if not tasks:
                return
            tasks.discard(task)
            if not tasks:
                self._tasks.pop(request_id, None)


@dataclass
class RequestTaskRegistry:
    manager: TaskManager
    request_id: str

    async def create_task(self, coro: Coroutine[Any, Any, Any]) -> TaskType:
        return await self.manager.create_task(self.request_id, coro)

    async def register_task(self, task: Optional[TaskType]) -> None:
        if task is None:
            return
        await self.manager.register_task(self.request_id, task)

    async def cancel_all(self, *, exclude: Optional[Set[TaskType]] = None) -> None:
        await self.manager.cancel_all(self.request_id, exclude=exclude)

    async def shutdown(self) -> None:
        await self.manager.clear(self.request_id)


task_manager = TaskManager()


async def provide_request_task_registry(request: Request) -> AsyncIterator[RequestTaskRegistry]:
    """FastAPI dependency yielding the per-request task registry."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    registry = RequestTaskRegistry(manager=task_manager, request_id=request_id)
    disconnect_task = await registry.create_task(
        task_manager.watch_disconnect(request_id, request),
    )

    try:
        yield registry
    finally:
        await registry.cancel_all(exclude={disconnect_task})
        disconnect_task.cancel()
        with suppress(asyncio.CancelledError):
            await disconnect_task
        await registry.shutdown()
