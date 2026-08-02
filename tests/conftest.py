from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

for _variable in (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

import langsmith  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

_DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="opentulpa-test")
_ORIGINAL_RUN_IN_EXECUTOR = asyncio.BaseEventLoop.run_in_executor


def _bounded_run_in_executor(
    self: asyncio.BaseEventLoop,
    executor: Any,
    func: Any,
    *args: Any,
) -> asyncio.Future[Any]:
    if executor is None:
        executor = _DEFAULT_EXECUTOR
    return _ORIGINAL_RUN_IN_EXECUTOR(self, executor, func, *args)


class _BoundedEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = super().new_event_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        return loop


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    return _BoundedEventLoopPolicy()


async def _inline_langsmith_aio_to_thread(
    default_aio_to_thread: Any,
    ctx: Any,
    func: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    del default_aio_to_thread
    return ctx.run(func, *args, **kwargs)


def pytest_configure() -> None:
    with suppress(RuntimeError, ValueError):
        threading.stack_size(1024 * 1024)
    asyncio.BaseEventLoop.run_in_executor = _bounded_run_in_executor
    langsmith.set_runtime_overrides(aio_to_thread=_inline_langsmith_aio_to_thread)


def pytest_unconfigure() -> None:
    asyncio.BaseEventLoop.run_in_executor = _ORIGINAL_RUN_IN_EXECUTOR
    _DEFAULT_EXECUTOR.shutdown(wait=True, cancel_futures=True)
    langsmith.set_runtime_overrides()


@pytest_asyncio.fixture(autouse=True)
async def _shutdown_default_executor_between_async_tests() -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(executor)
    try:
        yield
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
