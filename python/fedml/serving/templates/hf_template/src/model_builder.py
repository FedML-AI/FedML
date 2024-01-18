from copy import deepcopy
from dataclasses import dataclass, field, fields
import importlib.util
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

from accelerate.utils import compare_versions
import torch.cuda
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    pipeline as hf_pipeline,
)

from .constants import DEFAULT_MODEL_KWARGS, DEFAULT_HF_PIPELINE_KWARGS, MODEL_DTYPES, MODEL_NAMES
from .integrations import is_flash_attn_available, is_jinja2_available
from .modeling_utils import get_max_seq_length, get_model_class_from_config
from .typing import PathType, PipelineType
from .utils import get_real_path, is_directory, is_file, is_jinja_template


def get_class_in_module(class_name: str, module_path: PathType) -> Any:
    # Adapted from transformers.dynamic_module_utils.get_class_in_module
    # see https://stackoverflow.com/a/67692
    # see https://www.geeksforgeeks.org/how-to-import-a-python-module-given-the-full-path/
    assert Path(module_path).suffix.lower() == ".py", f"Only \".py\" files are supported."

    module_spec = importlib.util.spec_from_file_location("instruct_pipeline", str(module_path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return getattr(module, class_name)


def get_hf_pipeline(
        task: str,
        model_name_or_path: PathType,
        trust_remote_code: Optional[bool] = None,
        model_kwargs: Dict[str, Any] = None,
        pipeline_class: Optional[Any] = None,
        **pipeline_kwargs: Any
) -> PipelineType:
    if model_kwargs is None:
        model_kwargs = {}
    _model_kwargs = deepcopy(DEFAULT_MODEL_KWARGS)
    _model_kwargs.update(model_kwargs)
    # overwrite trust_remote_code if exist
    _model_kwargs["trust_remote_code"] = trust_remote_code

    _pipeline_kwargs = deepcopy(DEFAULT_HF_PIPELINE_KWARGS)
    _pipeline_kwargs.update(pipeline_kwargs)

    if is_directory(model_name_or_path):
        # build model; this is required for local model
        config = AutoConfig.from_pretrained(model_name_or_path, **_model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **_model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **_model_kwargs)

        if pipeline_class is None and hasattr(config, "custom_pipelines") and task in config.custom_pipelines:
            # infer pipeline_class if the target model has corresponding custom pipeline
            if not trust_remote_code:
                raise ValueError(
                    "Loading this pipeline requires you to execute the code in the pipeline file in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use,"
                    " then set the option `trust_remote_code=True` to remove this error."
                )

            class_reference = config.custom_pipelines[task]["impl"]

            module_file, class_name = class_reference.split(".")
            pipeline_class = get_class_in_module(class_name, Path(model_name_or_path) / f"{module_file}.py")

    else:
        config = None
        model = str(model_name_or_path)
        tokenizer = None

    # pipeline_kwargs and model_kwargs cannot both have trust_remote_code
    _model_kwargs.pop("trust_remote_code", trust_remote_code)

    return hf_pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        config=config,
        trust_remote_code=trust_remote_code,
        model_kwargs=_model_kwargs,
        pipeline_class=pipeline_class,
        **pipeline_kwargs
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Model name or path."})
    model_dtype: str = field(
        default="auto",
        metadata={
            "help": "Model data type. Set to \"auto\" to automatically infer the proper data type.",
            "choices": MODEL_DTYPES,
        }
    )
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum numbers of tokens to generate. Set to None or a non-positive value to automatically"
                    " infer it.",
        }
    )
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "The minimum numbers of tokens to generate, Set to None or a negative value to automatically"
                    " infer it.",
        }
    )
    do_sample: bool = field(
        default=DEFAULT_HF_PIPELINE_KWARGS["do_sample"],
        metadata={"help": "Whether to use sampling. If \"False\", use greedy decoding otherwise."}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    temperature: float = field(
        default=DEFAULT_HF_PIPELINE_KWARGS["temperature"],
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_k: int = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."}
    )
    top_p: float = field(
        default=1.,
        metadata={
            "help": "If < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p`"
                    " or higher are kept for generation.",
        }
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Jinja style chat template string or path to chat template file. Set to empty string "
                    "or None to use the default chat template.",
        }
    )
    default_chat_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Jinja style chat template string or path to chat template file. This template is used when model"
                    " specific template does not exist. Set to empty string or None to use the Hugging Face default"
                    " chat template.",
        }
    )
    verbose: bool = field(default=False, metadata={"help": "Whether to enable verbose mode for debugging."})
    auth_token: Optional[str] = field(
        default=None,
        metadata={"help": "Authentication token for Hugging Face private models such as Llama 2."}
    )
    strict_version_check: bool = field(
        default=True,
        metadata={"help": "Whether to enable strict dependency version check."}
    )
    strict_chat_template_check: bool = field(
        default=True,
        metadata={"help": "Whether to enable strict chat template check."}
    )

    def __post_init__(self) -> None:
        if self.auth_token is not None:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = str(self.auth_token)

        if is_file(self.model_name_or_path):
            raise ValueError(
                f"`model_name_or_path` must be a valid directory path or a valid hugging face model ID"
                f" but received a file path \"{self.model_name_or_path}\"."
            )

        elif is_directory(self.model_name_or_path):
            self.model_name_or_path = get_real_path(self.model_name_or_path)

        elif self.model_name_or_path not in MODEL_NAMES:
            # if model_name_or_path is not a local directory
            warnings.warn(
                f"`model_name_or_path` received an unverified model ID \"{self.model_name_or_path}\"."
                f" You may experience unexpected behavior from the model. Verified models are '{MODEL_NAMES}'."
            )

        config = AutoConfig.from_pretrained(self.model_name_or_path)
        required_transformers_version = getattr(config, "transformers_version", None)
        if (
                required_transformers_version is not None and
                compare_versions("transformers", "<", required_transformers_version)
        ):
            if self.strict_version_check:
                raise RuntimeError(
                    f"{self.model_name_or_path} requires `transformers` >= {required_transformers_version}."
                    f" This verification can be disabled by setting `strict_version_check` to \"False\" but"
                    f" you may experience unexpected behavior from the model."
                )
            else:
                warnings.warn(
                    f"{self.model_name_or_path} requires `transformers` >= {required_transformers_version}."
                    f" You may experience unexpected behavior from the model."
                )

        # get model context length
        model_context_length = get_max_seq_length(config)

        if self.max_new_tokens is None or self.max_new_tokens <= 0:
            # auto infer
            if model_context_length is not None:
                self.max_new_tokens = max(math.ceil(model_context_length / 2), 1)
            else:
                self.max_new_tokens = DEFAULT_HF_PIPELINE_KWARGS["max_new_tokens"]
            self.max_new_tokens = int(self.max_new_tokens)

        if self.min_new_tokens is not None and self.min_new_tokens < 0:
            self.min_new_tokens = None

        self.chat_template = self._parse_chat_template(self.chat_template)
        self.default_chat_template = self._parse_chat_template(self.default_chat_template)

        # update configs
        _ = self.torch_dtype
        _ = self.model_kwargs

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        if self.model_dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.model_dtype = "bf16"
            else:
                self.model_dtype = "fp16"

        torch_dtype = self.torch_dtype
        if self.model_dtype in ("fp32", "fp16", "bf16"):
            model_kwargs = dict(torch_dtype=torch_dtype)
        elif self.model_dtype == "4bit":
            model_kwargs = dict(
                torch_dtype=torch_dtype,
                bnb_4bit_compute_dtype=torch_dtype,
                load_in_4bit=True,
            )
        elif self.model_dtype == "8bit":
            model_kwargs = dict(
                torch_dtype=torch_dtype,
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"\"{self.model_dtype}\" is not a supported \"model_dtype\".")

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path=self.model_name_or_path, **model_kwargs)
        model_cls = get_model_class_from_config(config, **model_kwargs)
        model_supports_flash_attn_2 = getattr(model_cls, "_supports_flash_attn_2", False)

        if (
                compare_versions("transformers", ">=", "4.34.0")
                and torch.cuda.device_count() > 0
                and torch_dtype in (torch.bfloat16, torch.float16)
                and model_supports_flash_attn_2
                and is_flash_attn_available()
        ):
            if compare_versions("transformers", ">=", "4.36.0"):
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                model_kwargs["use_flash_attention_2"] = is_flash_attn_available()

        return model_kwargs

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.model_dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.model_dtype = "bf16"
            else:
                self.model_dtype = "fp16"

        if self.model_dtype in ("fp32", "8bit"):
            return torch.float32

        elif self.model_dtype == "fp16":
            return torch.float16

        elif self.model_dtype == "bf16":
            if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
                warnings.warn("Your device doesn't support bfloat16. Fall back to float16.")
                self.model_dtype = "fp16"
                return torch.float16

            return torch.bfloat16

        elif self.model_dtype == "4bit":
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        else:
            raise ValueError(f"\"{self.model_dtype}\" is not a supported \"model_dtype\".")

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        return dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

    def get_hf_pipeline(self, **kwargs) -> PipelineType:
        pipeline_kwargs = dict(
            model_name_or_path=self.model_name_or_path,
            model_kwargs=self.model_kwargs,
            **self.generation_kwargs,
        )
        pipeline_kwargs["model_kwargs"].update(kwargs.pop("model_kwargs", {}))
        pipeline_kwargs.update(kwargs)

        return get_hf_pipeline(**pipeline_kwargs)

    @classmethod
    def from_environ(cls, default_kwargs: Dict[str, Any] = None, **kwargs: Any) -> "ModelArguments":
        _kwargs = default_kwargs.copy() if default_kwargs is not None else {}
        _kwargs.update(kwargs)  # overwrite default kwargs
        _kwargs.update({
            f.name: os.environ[f.name.upper()] for f in fields(cls)
            if f.init and f.name.upper() in os.environ
        })

        # convert to sys.argv format
        args = [
            arg_str
            for k, v in _kwargs.items()
            for arg_str in (f"--{k}", str(v))
        ]

        parser = HfArgumentParser((cls,))
        return parser.parse_args_into_dataclasses(args=args)[0]

    def _parse_chat_template(self, chat_template: Optional[str]) -> Optional[str]:
        if not bool(chat_template):
            # empty string or None
            return None

        if not is_jinja2_available():
            if self.strict_chat_template_check:
                raise RuntimeError("Jinja2 >= 3.0.0 is required for customized chat templates.")
            else:
                warnings.warn(
                    "Jinja2 >= 3.0.0 is required for customized chat templates. Fallback to default chat template."
                )
                return None

        if is_file(chat_template):
            chat_template_file = chat_template

            with open(chat_template_file, "r") as f:
                chat_template_lines = f.readlines()
            chat_template = "".join(s.strip() for s in chat_template_lines)
        else:
            chat_template_file = None
            chat_template = chat_template

        if not is_jinja_template(chat_template):
            if chat_template_file is not None:
                error_msg = (
                    f"\"{chat_template_file}\" is not a valid chat template file. Chat template file must contain"
                    f" a valid jinja template string."
                )
            else:
                error_msg = (
                    "`chat_template` does not contain a valid chat template string. Chat template string must be"
                    " a valid jinja template string."
                )

            if self.strict_chat_template_check:
                raise ValueError(error_msg)
            else:
                warnings.warn(f"{error_msg} Fallback to default chat template.")
                chat_template = None

        return chat_template
