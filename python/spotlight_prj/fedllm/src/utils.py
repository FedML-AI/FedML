from typing import Any, Callable, TypeVar

import inspect
import logging

T = TypeVar("T")


def log_helper(
        message: Any,
        prefix: str = "",
        suffix: str = "",
        stack_prefix: str = "",
        stack_level: int = 1,
        level: int = logging.INFO
) -> None:
    logging.log(
        level=level,
        msg=f"{prefix} [{stack_prefix}{inspect.stack()[stack_level][3]}]: {message} {suffix}",
    )


def dummy_func(*args: Any, **kwargs: Any) -> None:
    return None


def get_dummy_func(default: T) -> Callable[..., T]:
    def _func(*args: Any, **kwargs: Any) -> T:
        return default

    return _func
