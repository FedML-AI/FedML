from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from argparse import Namespace
from copy import deepcopy
import inspect
import logging

from fedml.train.llm.integrations import is_fedml_available
from fedml.train.llm.utils import parse_hf_args as _parse_hf_args

if is_fedml_available():
    from fedml.arguments import Arguments

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


def parse_hf_args(
        dataclass_types: Union[Type[T], Iterable[Type[T]]],
        args: Optional[Union[Sequence[str], Namespace, Dict[str, Any], "Arguments"]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
) -> Tuple[T, ...]:
    if is_fedml_available() and isinstance(args, Arguments):
        _args = deepcopy(args)
        setattr(_args, "unitedllm_rank", args.rank)

        args = _args

    return _parse_hf_args(dataclass_types, args, parser_kwargs, **kwargs)
