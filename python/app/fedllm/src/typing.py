from typing import TypeVar, Union

from os import PathLike

from peft import PeftModel
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizerBase

__all__ = [
    "DataCollatorType",
    "ModelType",
    "PathType",
    "TokenizerType",
]

PathType = Union[str, PathLike]

DataCollatorType = TypeVar("DataCollatorType", bound=DataCollatorForLanguageModeling)
ModelType = TypeVar("ModelType", bound=Union[PreTrainedModel, PeftModel])
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)
