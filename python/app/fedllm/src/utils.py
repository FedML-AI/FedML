from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    TypeVar,
    Union,
)

from collections import OrderedDict
import inspect
import logging
from os import PathLike
from pathlib import Path

import torch
from torch import distributed as dist, Tensor
from torch.nn import Module
from transformers import PreTrainedModel, Trainer
from transformers.deepspeed import is_deepspeed_available, is_deepspeed_zero3_enabled
from peft import PeftModel, PromptLearningConfig

if is_deepspeed_available():
    import deepspeed

    is_deepspeed_initialized = deepspeed.comm.is_initialized
else:
    def is_deepspeed_initialized() -> bool:
        return False

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


def barrier() -> None:
    if is_deepspeed_initialized():
        deepspeed.comm.barrier()
    elif dist.is_initialized():
        dist.barrier()


def log_helper(
        message: str,
        prefix: str = "",
        suffix: str = "",
        stack_prefix: str = "",
        stack_level: int = 1,
        level: int = logging.INFO
):
    logging.log(
        level=level,
        msg=f"{prefix} [{stack_prefix}{inspect.stack()[stack_level][3]}]: {message} {suffix}",
    )


def is_deepspeed_module(model: Module) -> bool:
    # TODO: verify
    return any(hasattr(p, "ds_numel") for n, p in model.named_parameters())


def load_state_dict(
        model: Module,
        state_dict: Dict[str, Any],
        strict: bool = True,
        force_recursive_load: bool = False
) -> None:
    if (is_deepspeed_initialized() and is_deepspeed_zero3_enabled() and is_deepspeed_module(model)) \
            or force_recursive_load:
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        error_msgs = []
        load_state_dict_helper(
            module=model,
            state_dict=state_dict,
            error_msgs=error_msgs,
            strict=strict,
            metadata=metadata
        )
    else:
        model.load_state_dict(state_dict, strict=strict)


def load_state_dict_helper(
        module: Module,
        state_dict: Dict[str, Any],
        prefix: str = "",
        strict: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        error_msgs: Optional[List[str]] = None
) -> None:
    """
    Recursively load state_dict in to module. This function handles the partitioned cases when using DeepSpeed
        Stage 3 (zero3). This function is adapted from
        `transformers.modeling_utils._load_state_dict_into_model`; see
        https://github.com/huggingface/transformers/blob/539e2281cd97c35ef4122757f26c88f44115fa94/src/transformers/modeling_utils.py#LL493C25-L493C25

    Args:
        module: module (`torch.nn.Module`) to load state_dict.
        state_dict: a dict containing parameters and persistent buffers.
        prefix: the prefix for parameters and buffers used in this module.
        strict: whether to strictly enforce that the keys in `state_dict` with `prefix` match the names of
                parameters and buffers in this module.
        metadata: a dict containing the metadata for this module.
        error_msgs: error messages should be added to this list.

    Returns:

    """
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    if error_msgs is None:
        error_msgs = []

    # Parameters of module and children will start with prefix. We can exit early if there are none in this state_dict
    if any(key.startswith(prefix) for key in state_dict):
        if is_deepspeed_initialized() and is_deepspeed_zero3_enabled() and is_deepspeed_module(module):
            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context manager gathers the params of
                # the current layer, then loads from the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        module._load_from_state_dict(
                            state_dict=state_dict,
                            prefix=prefix,
                            local_metadata=local_metadata,
                            strict=strict,
                            missing_keys=[],
                            unexpected_keys=[],
                            error_msgs=error_msgs
                        )
        else:
            module._load_from_state_dict(
                state_dict=state_dict,
                prefix=prefix,
                local_metadata=local_metadata,
                strict=strict,
                missing_keys=[],
                unexpected_keys=[],
                error_msgs=error_msgs
            )

    for name, child in module.named_children():
        load_state_dict_helper(
            module=child,
            state_dict=state_dict,
            error_msgs=error_msgs,
            strict=strict,
            prefix=f"{prefix}{name}.",
            metadata=metadata
        )
