from typing import Any, TypeVar, Union

from os import PathLike

from peft import PeftModel, PeftConfig
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
)

__all__ = [
    "DataCollatorType",
    "is_model_config_type",
    "is_model_type",
    "ModelConfigType",
    "ModelType",
    "PathType",
    "TokenizerType",
]

PathType = Union[str, PathLike]

DataCollatorType = TypeVar("DataCollatorType", bound=DataCollatorForLanguageModeling)
ModelConfigType = TypeVar("ModelConfigType", bound=Union[PretrainedConfig, PeftConfig])
ModelType = TypeVar("ModelType", bound=Union[PreTrainedModel, PeftModel])
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def is_model_type(obj: Any) -> bool:
    return isinstance(obj, ModelType.__bound__.__args__)


def is_model_config_type(config: Any) -> bool:
    return isinstance(config, ModelConfigType.__bound__.__args__)
