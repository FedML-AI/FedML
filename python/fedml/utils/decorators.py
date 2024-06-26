from typing import Callable, Awaitable, TypeVar
import os
import time
import functools

T = TypeVar('T')


def timeit(func: Callable):
    """Print the runtime of the decorated function"""
    functools.wraps(func)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        run_time = end - start
        print(f"Finished {func.__name__!r} in {run_time:.4f} seconds")
        return value

    return wrapper


def async_timeit(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Print the runtime of the decorated async function"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = await func(*args, **kwargs)
        end = time.perf_counter()
        run_time = end - start
        print(f"Finished {func.__name__!r} in {run_time:.4f} seconds")
        return value

    return wrapper
