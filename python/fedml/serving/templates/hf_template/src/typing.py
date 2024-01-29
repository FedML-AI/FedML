from os import PathLike
from typing import Any, TypeVar, Union

from transformers import (
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PretrainedConfig,
)

__all__ = [
    "is_model_config_type",
    "is_model_type",
    "ModelConfigType",
    "ModelType",
    "PathType",
    "PipelineType",
    "str_to_bool",
    "TokenizerType",
]

PathType = Union[str, PathLike]

ModelConfigType = TypeVar("ModelConfigType", bound=PretrainedConfig)
ModelType = TypeVar("ModelType", bound=PreTrainedModel)
TokenizerType = TypeVar(
    "TokenizerType",
    bound=Union[PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast]
)
PipelineType = TypeVar("PipelineType", bound=Pipeline)


def is_model_type(obj: Any) -> bool:
    return isinstance(obj, ModelType.__bound__)


def is_model_config_type(config: Any) -> bool:
    return isinstance(config, ModelConfigType.__bound__)


def str_to_bool(s: str) -> bool:
    return False if s in ("false", "False") else bool(s)
