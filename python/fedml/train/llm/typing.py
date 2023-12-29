from typing import Any, Optional, TypeVar, Union

from os import PathLike

from datasets import Dataset, IterableDataset
from peft import PeftModel, PeftConfig
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
)

__all__ = [
    "DataCollatorType",
    "DatasetType",
    "is_model_config_type",
    "is_model_type",
    "LrSchedulerType",
    "ModelConfigType",
    "ModelType",
    "OptimizerType",
    "PathType",
    "PeftConfigType",
    "to_torch_dtype",
    "TokenizerType",
]

PathType = Union[str, PathLike]

DataCollatorType = TypeVar("DataCollatorType", bound=DataCollatorForLanguageModeling)
DatasetType = Union[Dataset, IterableDataset]

ModelConfigType = TypeVar("ModelConfigType", bound=Union[PretrainedConfig, PeftConfig])
ModelType = TypeVar("ModelType", bound=Union[PreTrainedModel, PeftModel])
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)
PeftConfigType = TypeVar("PeftConfigType", bound=PeftConfig)

OptimizerType = Optimizer
LrSchedulerType = Union[LambdaLR, ReduceLROnPlateau]

TORCH_DTYPE_ALIAS_MAPPING = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "float": "float32",
}


def is_model_type(obj: Any) -> bool:
    return isinstance(obj, ModelType.__bound__.__args__)


def is_model_config_type(config: Any) -> bool:
    return isinstance(config, ModelConfigType.__bound__.__args__)


def to_torch_dtype(torch_dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if isinstance(torch_dtype, torch.dtype) or torch_dtype is None:
        return torch_dtype

    elif isinstance(torch_dtype, str):
        torch_dtype = TORCH_DTYPE_ALIAS_MAPPING.get(torch_dtype, torch_dtype)
        return getattr(torch, torch_dtype)

    else:
        raise TypeError(f"Cannot convert object of type \"{type(torch_dtype)}\" to torch.dtype.")
