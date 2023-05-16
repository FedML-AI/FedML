from typing import TypeVar, MutableMapping, Union

from collections import OrderedDict
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel
from peft import PeftModel

T = TypeVar("T")


def to_device(data: T, device: Union[torch.device, str], non_blocking: bool = True) -> T:
    if isinstance(data, list):
        data = [to_device(d, device, non_blocking) for d in data]

    elif isinstance(data, tuple):
        data = tuple(to_device(d, device, non_blocking) for d in data)

    elif isinstance(data, MutableMapping):
        for k in data.keys():
            data[k] = to_device(data[k], device, non_blocking)

    elif isinstance(data, (Tensor, Module)):
        data = data.to(device, non_blocking=non_blocking)

    return data


def get_device(inputs: Union[Tensor, Module]) -> torch.device:
    if hasattr(inputs, "device"):
        return inputs.device
    else:
        return next(inputs.parameters()).device


def process_state_dict(state_dict: dict, reference_state_dict: dict) -> OrderedDict:
    output_state_dict = OrderedDict(state_dict)

    for k in state_dict.keys():
        if k not in reference_state_dict:
            output_state_dict.pop(k, None)
            continue

        v = state_dict[k]
        rv = reference_state_dict[k]

        if isinstance(v, Tensor) != isinstance(rv, Tensor):
            # if v and rv have different type
            output_state_dict.pop(k, None)

        if isinstance(v, Tensor) and isinstance(rv, Tensor) and v.shape != rv.shape:
            # v and rv have different shape
            output_state_dict.pop(k, None)

    return output_state_dict


def save_config(model: Union[PreTrainedModel, PeftModel], output_dir: PathLike) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        model.get_base_model().config.save_pretrained(output_dir)

    model.config.save_pretrained(output_dir)
