from typing import TypeVar, Union

from peft import PeftModel
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, PreTrainedModel

__all__ = [
    "DataCollatorType",
    "ModelType",
    "TokenizerType",
]

DataCollatorType = TypeVar("DataCollatorType", bound=DataCollatorForLanguageModeling)
ModelType = TypeVar("ModelType", bound=Union[PreTrainedModel, PeftModel])
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)
