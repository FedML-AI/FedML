from typing import Any, Dict, List, Optional, Union

from dataclasses import dataclass, field
import warnings

import numpy as np
from transformers import AutoConfig, DataCollatorForLanguageModeling

from .constants import IGNORE_INDEX
from .typing import (
    DataCollatorType,
    is_model_config_type,
    is_model_type,
    ModelConfigType,
    ModelType,
    TokenizerType,
)


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    escape_token: Optional[str] = field(
        default=None,
        metadata={"help": "If not `None`, will turn off loss for all tokens up until this token"}
    )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.escape_token is not None:
            # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
            # sequence of tokens. This should just be a single token.
            response_token_ids = self.tokenizer.encode(self.escape_token)

            labels = batch["labels"].clone()

            for i in range(len(examples)):
                for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                    response_token_ids_start_idx = idx
                    break
                else:
                    warnings.warn(
                        f"{type(self).__name__} Could not find response key {response_token_ids} in token IDs {batch['labels'][i]}"
                    )

                    response_token_ids_start_idx = len(batch["labels"][i])

                response_token_ids_end_idx = response_token_ids_start_idx + 1

                # Make pytorch loss function ignore all tokens up through the end of the response key
                labels[i, :response_token_ids_end_idx] = IGNORE_INDEX

            batch["labels"] = labels

        return batch


def get_data_collator(
        tokenizer: TokenizerType,
        escape_token: Optional[str] = None,
        pad_to_multiple_of: Optional[int] = 8,
        **kwargs: Any
) -> DataCollatorType:
    _kwargs = dict(
        mlm=False,
        return_tensors="pt"
    )
    _kwargs.update(**kwargs)

    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of,
        escape_token=escape_token,
        **_kwargs
    )


def get_max_seq_length(model_or_config: Union[str, ModelConfigType, ModelType], **kwargs: Any) -> Optional[int]:
    if is_model_config_type(model_or_config):
        config = model_or_config
    elif is_model_type(model_or_config):
        config = model_or_config.config
    elif isinstance(model_or_config, str):
        config = AutoConfig.from_pretrained(model_or_config, **kwargs)
    else:
        raise TypeError(f"\"{type(model_or_config)}\" is not a supported model_or_config type.")

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        embedding_size = getattr(config, length_setting, None)
        if embedding_size is not None:
            return embedding_size
    else:
        return None


def get_vocab_size(model_or_config: Union[str, ModelConfigType, ModelType], **kwargs: Any) -> Optional[int]:
    if is_model_config_type(model_or_config):
        config = model_or_config
    elif is_model_type(model_or_config):
        config = model_or_config.config
    elif isinstance(model_or_config, str):
        config = AutoConfig.from_pretrained(model_or_config, **kwargs)
    else:
        raise TypeError(f"\"{type(model_or_config)}\" is not a supported model_or_config type.")

    return getattr(config, "vocab_size", None)
