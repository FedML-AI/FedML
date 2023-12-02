from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from argparse import Namespace
from copy import copy
from dataclasses import fields
import os
from pathlib import Path
import shutil

import torch.cuda
from torch import Tensor
from transformers import HfArgumentParser
from peft import PeftModel, PromptLearningConfig

from .integrations import is_fedml_available
from .typing import ModelType, PathType

if is_fedml_available():
    from fedml.arguments import Arguments

T = TypeVar("T")
M = TypeVar("M", bound=MutableMapping)


def get_real_path(path: PathType) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def is_file(path: PathType) -> bool:
    return os.path.isfile(get_real_path(path))


def is_directory(path: PathType) -> bool:
    return os.path.isdir(get_real_path(path))


def move_directory_content(src_path: PathType, dest_path: PathType) -> None:
    """
    Move all files/subdirectories in src_path into dest_path then remove src_path.

    Args:
        src_path: source directory path
        dest_path: destination directory path

    Returns:

    """
    if not is_directory(src_path):
        raise FileNotFoundError(f"\"{src_path}\" is not a directory.")
    if is_file(dest_path):
        raise FileExistsError(f"\"{dest_path}\" is an existing file.")

    if get_real_path(src_path) == get_real_path(dest_path):
        return

    src_path = Path(src_path)
    dest_path = Path(dest_path)

    for p in tuple(src_path.iterdir()):
        shutil.move(str(p), str(dest_path / p.relative_to(src_path)))
    shutil.rmtree(str(src_path))


def save_config(model: ModelType, output_dir: PathType) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        """
        adapted from peft.PeftModel.save_pretrained()
        """
        peft_model = model
        model = peft_model.get_base_model()

        for adapter_name, peft_config in peft_model.peft_config.items():
            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    peft_model.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig)
                    else peft_model.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(str(output_dir))
            peft_config.inference_mode = inference_mode

    model.config.save_pretrained(str(output_dir))


def parse_hf_args(
        dataclass_types: Union[Type[T], Iterable[Type[T]]],
        args: Optional[Union[Sequence[str], Namespace, Dict[str, Any], "Arguments"]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
) -> Tuple[T, ...]:
    if parser_kwargs is None:
        parser_kwargs = {}

    parser = HfArgumentParser(dataclass_types, **parser_kwargs)

    if args is None or isinstance(args, Sequence):
        return parser.parse_args_into_dataclasses(args=args, **kwargs)

    elif isinstance(args, Namespace):
        args_dict = dict(args.__dict__)

    elif isinstance(args, dict):
        args_dict = args

    elif is_fedml_available() and isinstance(args, Arguments):
        args_dict = dict(args.__dict__)
        if not getattr(args, "using_gpu", True) or torch.cuda.device_count() == 1:
            # If not using GPU or not having more than one GPUs
            args_dict.pop("local_rank", None)
            args_dict.pop("device", None)

        if "client_optimizer" in args_dict:
            # if contains `client_optimizer` and `optim` is not set, use `client_optimizer`
            args_dict.setdefault("optim", args_dict["client_optimizer"])

    else:
        raise TypeError(f"{type(args)} is not a supported type")

    kwargs.setdefault("allow_extra_keys", True)
    return parser.parse_dict(args_dict, **kwargs)


# Adapted from `transformers.training_args.TrainingArguments.to_dict`
def dataclass_to_dict(dataclass_obj) -> Dict[str, Any]:
    d = {f.name: getattr(dataclass_obj, f.name) for f in fields(dataclass_obj) if f.init}

    for k, v in d.items():
        if k.endswith("_token"):
            d[k] = f"<{k.upper()}>"

    return d


# Adapted from `transformers.training_args.TrainingArguments.to_sanitized_dict`
def dataclass_to_sanitized_dict(dataclass_obj) -> Dict[str, Any]:
    return to_sanitized_dict(dataclass_to_dict(dataclass_obj))


def to_sanitized_dict(d: Mapping[str, Any]) -> Dict[str, Any]:
    valid_types = [bool, int, float, str, Tensor]
    return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


def replace_if_exists(d: M, *mappings: Mapping, inplace: bool = False, **kwargs: Any) -> M:
    if not inplace:
        d = copy(d)

    for m in mappings:
        if not isinstance(m, Mapping):
            raise TypeError(f"All positional inputs must be `Mapping` objects but received \"{type(m)}\" object.")

        d.update({k: v for k, v in m.items() if k in d})

    d.update({k: v for k, v in kwargs.items() if k in d})

    return d
