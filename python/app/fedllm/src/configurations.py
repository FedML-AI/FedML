from typing import List, Optional

from dataclasses import dataclass, field
import os
import warnings

from accelerate.utils import compare_versions
from transformers import TrainingArguments

from .constants import (
    DATASET_NAMES,
    FINETUNE_TASKS,
    MODEL_NAMES,
)
from .utils import is_directory, is_file


@dataclass
class FinetuningArguments(TrainingArguments):
    task: str = field(default="finetune", metadata={"help": "finetune task type", "choices": FINETUNE_TASKS})

    @property
    def is_instruction_finetune(self) -> bool:
        return self.task == "instruction"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="EleutherAI/pythia-70m", metadata={"help": "model name or path."})
    use_lora: bool = field(default=False, metadata={"help": "Set to `True` to enable LoRA."})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout."})
    lora_on_all_modules: bool = field(
        default=False,
        metadata={"help": "If `True`, apply LoRA on all supported layers and track gradient for all non-LoRA layers."}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
            "nargs": "+",
        },
    )
    auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Authentication token for Hugging Face private models such as Llama 2.",
        },
    )
    load_pretrained: bool = field(default=True, metadata={"help": "If `True`, load pretrained model."})

    def __post_init__(self) -> None:
        if is_file(self.model_name_or_path):
            if self.load_pretrained:
                raise ValueError(
                    "\"model_name_or_path\" must be a model ID or directory path if \"load_pretrained\" is `True`."
                )

        elif not is_directory(self.model_name_or_path):
            # if model_name_or_path is not a local file or directory
            if self.model_name_or_path not in MODEL_NAMES:
                model_names_str = "', '".join(MODEL_NAMES)

                raise ValueError(
                    f"\"model_name_or_path\" must be a valid file/directory path or a supported model ID ("
                    f"choose from '{model_names_str}') but received \"{self.model_name_or_path}\"."
                )

            if self.model_name_or_path.startswith("meta-llama/Llama-2-"):
                if compare_versions("transformers", "<", "4.31.0"):
                    raise NotImplementedError(f"{self.model_name_or_path} requires transformers >= 4.31.0")

                if self.auth_token is not None:
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = str(self.auth_token)

                # need to verify if already logged in
                from huggingface_hub import HfApi
                from huggingface_hub.utils import LocalTokenNotFoundError

                try:
                    HfApi().whoami()
                except LocalTokenNotFoundError:
                    raise LocalTokenNotFoundError(
                        f"Token is required for {self.model_name_or_path}, but no token found. You need to provide a"
                        f" token or be logged in to Hugging Face."
                        f"\nTo pass a token, you could pass `--auth_token \"<your token>\"` or set environment"
                        f" variable `HUGGING_FACE_HUB_TOKEN=\"${{your_token}}\"`."
                        f"\nTo login, use `huggingface-cli login` or `huggingface_hub.login`."
                        f" See https://huggingface.co/settings/tokens."
                    )


@dataclass
class DatasetArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "dataset name", "choices": DATASET_NAMES})
    dataset_path: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to the training data file(s).", "nargs": "+"}
    )
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "dataset configuration name"})
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "If `True`, streams the data progressively while iterating on the dataset."}
    )
    test_dataset_size: int = field(
        default=-1,
        metadata={"help": "test set size. Will be ignored if `dataset_path` has more than 1 entry."}
    )
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "max sequence length."})
    truncate_long_seq: bool = field(
        default=True,
        metadata={"help": "Whether to truncate long sequences whose length > max_seq_length."}
    )
    remove_long_seq: bool = field(
        default=False,
        metadata={"help": "Whether to remove all data whose token length > max_seq_length."}
    )

    def __post_init__(self) -> None:
        if self.dataset_name is not None:
            self.dataset_path = []

        elif len(self.dataset_path) <= 0:
            # if dataset_name is None
            raise ValueError("\"dataset_name\" and \"dataset_path\" cannot both be empty.")

        elif len(self.dataset_path) >= 3:
            warnings.warn("More than 2 dataset paths provided. Only the first 2 will be loaded.")
            self.dataset_path = self.dataset_path[:2]

        elif len(self.dataset_path) == 1 and self.test_dataset_size <= 0:
            raise ValueError("\"test_dataset_size\" must be a positive value when dataset_path has only 1 entry.")

        if self.remove_long_seq and not self.truncate_long_seq:
            warnings.warn("\"truncate_long_seq\" is set to `True` since \"remove_long_seq\" is `True`.")
            self.truncate_long_seq = True

    @property
    def truncation_max_length(self) -> Optional[int]:
        if self.max_seq_length is not None and self.remove_long_seq:
            # set to max_seq_length + 1 so that sequences has length >= max_seq_lengths can be filtered
            return self.max_seq_length + 1
        else:
            return self.max_seq_length
