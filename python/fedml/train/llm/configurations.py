from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import base64
from dataclasses import dataclass, field, is_dataclass
import os.path
import warnings

from accelerate.utils import compare_versions
from datasets import get_dataset_split_names
from peft import LoraConfig, TaskType
import torch
from transformers import AutoConfig, TrainingArguments
import yaml

from .constants import (
    CUSTOM_LOGGERS,
    DATASET_NAMES,
    MODEL_DTYPES,
    MODEL_DTYPE_MAPPING,
    MODEL_NAMES,
    PEFT_TYPES,
    PROMPT_STYLES,
)
from .dataset_utils import DEFAULT_COLUMN_NAME_MAPPING, DEFAULT_KEYWORD_REPLACEMENTS
from .integrations import is_transformers_greater_or_equal_4_34, is_transformers_greater_or_equal_4_36
from .modeling_utils import get_model_class_from_config
from .typing import ModelConfigType, ModelType, PeftConfigType, to_torch_dtype
from .utils import dataclass_to_dict, is_directory, is_file, to_sanitized_dict


@dataclass
class ExperimentArguments(TrainingArguments):
    custom_logger: List[str] = field(
        default_factory=list,
        metadata={
            "help": "The list of customized logger to report the results and logs to.",
            "choices": CUSTOM_LOGGERS,
            "nargs": "+",
        }
    )
    extra_save_steps: List[int] = field(
        default_factory=list,
        metadata={
            "help": "Extra steps to save checkpoints. Disabled if `save_strategy` is \"no\" or if contains"
                    " non-positive number.",
            "nargs": "+",
        }
    )
    # optional
    model_args: Optional["ModelArguments"] = field(
        default=None,
        init=False,
        metadata={
            "help": "Reference to the `ModelArguments` object. This should be added by calling "
                    "`add_and_verify_model_args`"
        }
    )
    dataset_args: Optional["DatasetArguments"] = field(
        default=None,
        init=False,
        metadata={
            "help": "Reference to the `DatasetArguments` object. This should be added by calling "
                    "`add_and_verify_dataset_args`"
        }
    )

    def __post_init__(self):
        if "none" in self.custom_logger:
            self.custom_logger = []
        elif "all" in self.custom_logger:
            self.custom_logger = [l for l in CUSTOM_LOGGERS if l not in ("all", "none")]

        if any(v <= 0 for v in self.extra_save_steps):
            self.extra_save_steps = []
        self.extra_save_steps = sorted(set(self.extra_save_steps))

        super().__post_init__()

        # for `transformers>=4.35.0`, `gradient_checkpointing_kwargs` is added to allow passing arguments directly to
        # `torch.utils.checkpoint.checkpoint`
        if hasattr(self, "gradient_checkpointing_kwargs"):
            if self.gradient_checkpointing and self.gradient_checkpointing_kwargs is None:
                # see https://pytorch.org/docs/stable/checkpoint.html
                # see https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d
                self.gradient_checkpointing_kwargs = dict(use_reentrant=True)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()

        if is_dataclass(self.model_args):
            d["model_args"] = dataclass_to_dict(self.model_args)

        if is_dataclass(self.dataset_args):
            d["dataset_args"] = dataclass_to_dict(self.dataset_args)

        return d

    def to_sanitized_dict(self) -> Dict[str, Any]:
        d = super().to_sanitized_dict()

        model_args_dict = d.get("model_args", None)
        if isinstance(model_args_dict, Mapping):
            d["model_args"] = to_sanitized_dict(model_args_dict)

        dataset_args_dict = d.get("dataset_args", None)
        if isinstance(dataset_args_dict, Mapping):
            d["dataset_args"] = to_sanitized_dict(dataset_args_dict)

        return d

    def add_and_verify_args(self, *args: Any) -> None:
        for args_obj in args:
            if isinstance(args_obj, ModelArguments):
                self.add_and_verify_model_args(args_obj)

            elif isinstance(args_obj, DatasetArguments):
                self.add_and_verify_dataset_args(args_obj)

            else:
                raise TypeError(f"{type(args_obj)} is not a supported args type.")

    def add_and_verify_model_args(self, model_args: "ModelArguments") -> None:
        self.model_args = model_args

    def add_and_verify_dataset_args(self, dataset_args: "DatasetArguments") -> None:
        if dataset_args.tokenize_on_the_fly and self.remove_unused_columns:
            # See https://github.com/huggingface/datasets/issues/1867
            warnings.warn(
                f"When tokenizing on-the-fly, need to disable `remove_unused_columns` and `group_by_length`"
                f" to ensure correctness and performance."
            )

            self.remove_unused_columns = False
            # see https://discuss.huggingface.co/t/set-transform-and-group-by-length-true/6666/2
            self.group_by_length = False

        self.dataset_args = dataset_args


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Model name or path."})
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Model repo revision. If set to empty string, will use the HEAD of the main branch."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name or path."}
    )
    tokenizer_revision: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer repo revision. If set to empty string, will use the HEAD of the main branch."}
    )
    model_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model data type. Set to \"none\" to use the default data type.",
            "choices": MODEL_DTYPES,
        }
    )
    peft_type: str = field(
        default="none",
        metadata={"help": "PEFT type. Set to \"none\" to disable PEFT.", "choices": PEFT_TYPES}
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_on_all_modules: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA on all supported layers."}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
            "nargs": "+",
        }
    )
    auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "Authentication token for Hugging Face private models such as Llama 2."}
    )
    load_pretrained: bool = field(default=True, metadata={"help": "Whether to load pretrained model weights."})
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow for custom code defined on Hugging Face Hub in their own modeling, configuration,"
                    " tokenization or even pipeline files. This option should only be set to `True` for repositories"
                    " you trust and in which you have read the code, as it will execute code present on the Hub on your"
                    " local machine.",
        }
    )
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention."})
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use the fast tokenizer from `tokenizers` library."},
    )
    # private args for easier inheritance
    _verified_model_names: Tuple[str] = field(
        default=tuple(MODEL_NAMES),
        init=False,
        metadata={"help": "The Hugging Face ID of the supported/verified models."}
    )

    def __post_init__(self) -> None:
        if self.auth_token is not None:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = str(self.auth_token)

        if not bool(self.model_revision):
            self.model_revision = None

        if not bool(self.tokenizer_name_or_path):
            self.tokenizer_name_or_path = None

        if not bool(self.tokenizer_revision):
            self.tokenizer_revision = None

        if is_file(self.model_name_or_path):
            raise ValueError(
                f"`model_name_or_path` must be a valid directory path or a valid hugging face model ID"
                f" but received a file path \"{self.model_name_or_path}\"."
            )

        elif is_directory(self.model_name_or_path):
            # only expand "~/" to full path
            self.model_name_or_path = os.path.expanduser(self.model_name_or_path)

        elif self.model_name_or_path not in self._verified_model_names:
            # if model_name_or_path is not a local directory
            warnings.warn(
                f"`model_name_or_path` received an unverified model ID \"{self.model_name_or_path}\"."
                f" You may experience unexpected behavior from the model. Verified models are"
                f" '{self._verified_model_names}'."
            )

        config = AutoConfig.from_pretrained(self.model_name_or_path, revision=self.model_revision)
        required_transformers_version = getattr(config, "transformers_version", None)
        if (
                required_transformers_version is not None and
                compare_versions("transformers", "<", required_transformers_version)
        ):
            raise RuntimeError(
                f"{self.model_name_or_path} requires `transformers` >= {required_transformers_version}"
            )

        if self.model_dtype is not None:
            # convert model_dtype to canonical name
            self.model_dtype = MODEL_DTYPE_MAPPING[self.model_dtype]

    @property
    def torch_dtype(self) -> Optional[torch.dtype]:
        return to_torch_dtype(self.model_dtype)

    def get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        model_kwargs = dict(
            pretrained_model_name_or_path=self.model_name_or_path,
            revision=self.model_revision,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs.update(kwargs)

        config: Optional[ModelConfigType] = model_kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(**model_kwargs)
        model_cls = get_model_class_from_config(config, **model_kwargs)

        # remove flash attention keys and later add relevant keys back
        attn_implementation = model_kwargs.pop("attn_implementation", None)

        if (
                model_kwargs.pop("use_flash_attention_2", None)
                or attn_implementation == "flash_attention_2"
                or self.use_flash_attention
        ):
            # if enable flash attention
            if getattr(model_cls, "_supports_flash_attn_2", False):
                # for `transformers >= 4.34.0`, some huggingface models natively support flash attention v2.
                # only set `use_flash_attention_2` or `attn_implementation` for supported models
                # see https://github.com/huggingface/transformers/issues/26350

                if is_transformers_greater_or_equal_4_36():
                    # `transformers >= 4.36.0` updated flash attention API
                    model_kwargs["attn_implementation"] = attn_implementation = "flash_attention_2"
                elif is_transformers_greater_or_equal_4_34():
                    # if `4.34.0 <= transformers < 4.36.0`
                    model_kwargs["use_flash_attention_2"] = True
                    attn_implementation = None

                if not self.load_pretrained:
                    # see https://discuss.huggingface.co/t/how-to-load-model-without-pretrained-weight/34155/3
                    # When not loading pretrain weights, need to create model with `AutoModelForCausalLM.from_config`

                    # must rebuild config with the updated model kwargs
                    config: ModelConfigType = AutoConfig.from_pretrained(**model_kwargs)

                    if is_transformers_greater_or_equal_4_34() and not is_transformers_greater_or_equal_4_36():
                        # For `4.34.0 <= transformers < 4.36.0`, `AutoModel.from_config` does not support
                        # `use_flash_attention_2` flag. Need to enable manually
                        config = model_cls._check_and_enable_flash_attn_2(
                            config,
                            torch_dtype=model_kwargs["torch_dtype"],
                            device_map=model_kwargs.get("device_map", None)
                        )

                    model_kwargs["config"] = config

            else:
                if not is_transformers_greater_or_equal_4_34():
                    warnings.warn(f"`transformers >= 4.34.0` is required for native flash attention support.")
                else:
                    warnings.warn(
                        f"Model \"{model_kwargs['pretrained_model_name_or_path']}\" does not natively support flash"
                        f" attention. Fallback to default options."
                    )

                # disable flash attention flags
                attn_implementation = None

        # add back `attn_implementation` if needed
        if is_transformers_greater_or_equal_4_36() and attn_implementation != "flash_attention_2":
            model_kwargs["attn_implementation"] = attn_implementation

        return model_kwargs

    def get_tokenizer_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        tokenizer_kwargs = dict(
            pretrained_model_name_or_path=(
                self.tokenizer_name_or_path if bool(self.tokenizer_name_or_path) else self.model_name_or_path
            ),
            revision=self.tokenizer_revision if bool(self.tokenizer_revision) else self.model_revision,
            use_fast=self.use_fast_tokenizer,
            trust_remote_code=self.trust_remote_code,
        )
        tokenizer_kwargs.update(kwargs)

        return tokenizer_kwargs

    def get_peft_config(self, model: ModelType, **kwargs: Any) -> Optional[PeftConfigType]:
        peft_kwargs = dict(
            base_model_name_or_path=self.model_name_or_path,
            task_type=TaskType.CAUSAL_LM,
            revision=self.model_revision,
        )
        peft_kwargs.update(kwargs)

        if self.peft_type == "lora":
            if self.lora_on_all_modules:
                from fedml.train.llm.peft_utils import LORA_LAYER_TYPES

                additional_target_modules = []
                for n, m in model.named_modules():
                    if isinstance(m, tuple(LORA_LAYER_TYPES)):
                        additional_target_modules.append(n.split(".")[-1])

                if len(additional_target_modules) > 0:
                    if self.lora_target_modules is None:
                        self.lora_target_modules = []
                    self.lora_target_modules = list(set(self.lora_target_modules + additional_target_modules))

            return LoraConfig(
                r=self.lora_r,
                target_modules=self.lora_target_modules,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                **peft_kwargs
            )

        else:
            return None


@dataclass
class DatasetArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face dataset name. If set to an non-empty string, will override `dataset_path`."}
    )
    dataset_path: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Path to dataset file(s). If contains multiple entries, the 1st entry is considered"
                    " the training split and the 2nd entry is the test split.",
            "nargs": "+",
        }
    )
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "dataset configuration name"})
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the data progressively while iterating on the dataset."}
    )
    test_dataset_size: int = field(
        default=-1,
        metadata={
            "help": "The test dataset size. Will be ignored if set to a non-positive value, if `dataset_name`"
                    " contains \"test\" split, or if `dataset_path` has at least 2 entries.",
        }
    )
    test_dataset_ratio: Optional[float] = field(
        default=-1.0,
        metadata={
            "help": "Test dataset ratio. If set to a valid value (`0 < test_dataset_ratio < 1`) will override"
                    " `test_dataset_size`. Will be ignored if set to an invalid value, if `dataset_name`"
                    " contains \"test\" split, or if `dataset_path` has at least 2 entries.",
        }
    )
    eval_dataset_size: int = field(
        default=-1,
        metadata={
            "help": "The evaluation dataset size. This dataset is used to evaluate the model performance"
                    " during training. Set to a non-positive number to use the test dataset for the"
                    " evaluation during training.",
        }
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "The max sequence length. If unspecified, will use the context length of the model."})
    truncate_long_seq: bool = field(
        default=True,
        metadata={"help": "Whether to truncate long sequences whose length > max_seq_length."}
    )
    remove_long_seq: bool = field(
        default=False,
        metadata={"help": "Whether to remove all data whose token length > max_seq_length."}
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={"help": "Max number of processes when generating cache."}
    )
    tokenize_on_the_fly: bool = field(
        default=False,
        metadata={"help": "Whether to tokenize the input on-the-fly."}
    )
    prompt_style: str = field(
        default="default",
        metadata={"help": "Prompt template style.", "choices": PROMPT_STYLES}
    )
    response_template: str = field(
        default="",
        metadata={
            "help": f"The response template for instruction fine-tuning such as `### Response:`. If set to"
                    f" a non-empty string, the response template and all text before it will be excluded"
                    f" from the loss computation.",
        }
    )
    cleanup_data_cache: bool = field(
        default=False,
        metadata={
            "help": f"Whether to cleanup the data cache before data preprocessing. By default the `datasets`"
                    f" library caches preprocessed data on disk. When developing/changing the data preprocessing"
                    f" logic we need to clean the data cache to ensure the most up-to-date data is generated.",
        }
    )
    disable_data_keyword_replacement: bool = field(
        default=False,
        metadata={"help": f"Whether to disable the keyword replacement for data preprocessing."}
    )
    data_keyword_replacements: Optional[str] = field(
        default=None,
        metadata={
            "help": f"Dataset keyword replacement. This replaces keywords in dataset with desired text. Can be the"
                    f" path to a YAML file, a base64 encoded YAML string, or an already loaded YAML as a dict. Set"
                    f" to an empty string to use the default value.",
        }
    )
    column_name_mapping: Optional[str] = field(
        default=None,
        metadata={
            "help": f"Dataset column name mapping. Can be the path to a YAML file, a base64 encoded YAML string, or an"
                    f" already loaded YAML as a dict. Set to an empty string to use the default value.",
        }
    )
    # private args for easier inheritance
    _verified_dataset_names: Tuple[str] = field(
        default=tuple(DATASET_NAMES),
        init=False,
        metadata={"help": "The Hugging Face ID of the supported/verified datasets."}
    )

    def __post_init__(self) -> None:
        if not bool(self.dataset_name):
            # if `dataset_name` is None or empty string
            self.dataset_name = None

        if bool(self.dataset_name):
            # if `dataset_name` is a valid string
            self.dataset_path = []

            if self.dataset_name not in self._verified_dataset_names:
                warnings.warn(
                    f"`dataset_name` received an unverified dataset ID \"{self.dataset_name}\"."
                    f" Data processing may not work on this dataset. Verified datasets are"
                    f" '{self._verified_dataset_names}'."
                )

            split_names = get_dataset_split_names(self.dataset_name, self.dataset_config_name)
            if len(split_names) <= 1 and self.test_size is None:
                raise ValueError(
                    f"`{self.dataset_name}` only has 1 split. A positive `test_dataset_ratio`"
                    f" or `test_dataset_size` is required."
                )

        elif len(self.dataset_path) <= 0:
            # if dataset_name is None
            raise ValueError("`dataset_name` and `dataset_path` cannot both be empty.")

        elif len(self.dataset_path) > 2:
            warnings.warn("More than 2 dataset paths provided. Only the first 2 will be loaded.")
            self.dataset_path = self.dataset_path[:2]

        elif len(self.dataset_path) == 1 and self.test_size is None:
            raise ValueError(
                "A positive `test_dataset_ratio` or `test_dataset_size` is required when"
                " `dataset_path` has only 1 entry."
            )

        if self.dataset_streaming:
            if self.tokenize_on_the_fly:
                warnings.warn(
                    "`dataset_streaming` is not compatible with `tokenize_on_the_fly`."
                    " Setting `tokenize_on_the_fly` to \"False\"."
                )
                self.tokenize_on_the_fly = False

            if self.remove_long_seq:
                warnings.warn(
                    "`dataset_streaming` is not compatible with `remove_long_seq`."
                    " Setting `remove_long_seq` to \"False\"."
                )
                self.remove_long_seq = False

            if self.eval_dataset_size > 0:
                warnings.warn(
                    f"Using `dataset_streaming` with `eval_dataset_size={self.eval_dataset_size}`"
                    f" may slow down the data processing since this requires partially downloading the"
                    f" dataset."
                )

        if self.remove_long_seq and self.tokenize_on_the_fly:
            raise ValueError("`remove_long_seq` is not compatible with `tokenize_on_the_fly`")

        if self.remove_long_seq and not self.truncate_long_seq:
            warnings.warn("`truncate_long_seq` is set to \"True\" since `remove_long_seq` is \"True\".")
            self.truncate_long_seq = True

        if self.dataset_num_proc is not None:
            max_cpu_count = os.cpu_count()
            if max_cpu_count is None:
                warnings.warn(f"Cannot detect CPU number; fallback to 0.")
                max_cpu_count = 0

            if self.dataset_num_proc <= 0:
                warnings.warn("Received non-positive `dataset_num_proc`; fallback to CPU count.")
                self.dataset_num_proc = max_cpu_count

            # cap process number to CPU cores
            self.dataset_num_proc = min(self.dataset_num_proc, max_cpu_count)

        if len(self.response_template) > 0 and not self.response_template.endswith("\n"):
            # response_template should always end with newline
            self.response_template += "\n"

        self.data_keyword_replacements: Dict[str, str] = self._parse_data_keyword_replacements()
        self.column_name_mapping: Dict[str, str] = self._parse_column_name_mapping()

    @property
    def test_size(self) -> Optional[Union[int, float]]:
        if 0 < self.test_dataset_ratio < 1:
            return self.test_dataset_ratio

        elif self.test_dataset_size > 0:
            return self.test_dataset_size

        else:
            return None

    @property
    def truncation_max_length(self) -> Optional[int]:
        if self.max_seq_length is not None and self.remove_long_seq:
            # set to max_seq_length + 1 so that sequences with length >= max_seq_lengths can be
            # filtered out by removing all entries with length > max_seq_length.
            return self.max_seq_length + 1
        else:
            return self.max_seq_length

    def _parse_data_keyword_replacements(self) -> Dict[str, str]:
        if self.data_keyword_replacements is None or self.data_keyword_replacements == "":
            config = DEFAULT_KEYWORD_REPLACEMENTS.copy()
        else:
            config = self._parse_config(self.data_keyword_replacements)

        # `key == value` is meaningless and should be removed
        return {k: v for k, v in config.items() if k != v}

    def _parse_column_name_mapping(self) -> Dict[str, str]:
        if self.column_name_mapping is None or self.column_name_mapping == "":
            config = DEFAULT_COLUMN_NAME_MAPPING.copy()
        else:
            config = self._parse_config(self.column_name_mapping)

        # `key == value` is meaningless and should be removed
        return {k: v for k, v in config.items() if k != v}

    @staticmethod
    def _parse_config(config_or_str: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config_or_str, Mapping):
            config = dict(config_or_str)

        elif isinstance(config_or_str, str):
            if is_file(config_or_str):
                with open(config_or_str, "r") as f:
                    config: Dict[str, Any] = yaml.safe_load(f)

            else:
                config_decoded = base64.urlsafe_b64decode(config_or_str).decode("utf-8")
                config: Dict[str, Any] = yaml.safe_load(config_decoded)

        else:
            raise TypeError(f"\"{config_or_str}\" is not a supported config type.")

        return config
