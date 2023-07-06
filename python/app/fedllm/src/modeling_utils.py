from typing import Any, Dict, List, Optional, Union

from dataclasses import dataclass, field

import numpy as np
from transformers import DataCollatorForLanguageModeling

from .constants import IGNORE_INDEX
from .typing import DataCollatorType, TokenizerType


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
                response_token_ids_start_idx = None
                for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                    response_token_ids_start_idx = idx
                    break

                if response_token_ids_start_idx is None:
                    raise RuntimeError(
                        f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                    )

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
