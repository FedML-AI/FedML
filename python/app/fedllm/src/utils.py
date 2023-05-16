from typing import TypeVar, MutableMapping, Union

from collections import OrderedDict
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedModel, Trainer
from peft import PeftModel, PromptLearningConfig

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


def save_config(model: Union[PreTrainedModel, PeftModel], output_dir: Union[str, PathLike]) -> None:
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


def is_main_process(trainer: Trainer, local: bool = True) -> bool:
    return trainer.is_local_process_zero() if local else trainer.is_world_process_zero()


def should_process_save(trainer: Trainer) -> bool:
    return is_main_process(trainer, trainer.args.save_on_each_node)
