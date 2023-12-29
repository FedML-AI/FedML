from typing import Any, Dict, List, Optional

from fedml.train.llm.distributed import gather_parameter, get_rank, is_deepspeed_initialized, is_deepspeed_module
from fedml.train.llm.integrations import is_deepspeed_zero3_enabled
from torch.nn import Module
from torch.nn.modules.module import _IncompatibleKeys


def load_state_dict(
        model: Module,
        state_dict: Dict[str, Any],
        strict: bool = True,
        force_recursive_load: bool = False
) -> _IncompatibleKeys:
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
            strict=strict,
            metadata=metadata,
            error_msgs=error_msgs
        )

        model_keys = {n for n, _ in model.named_parameters()}
        model_keys.update(n for n, _ in model.named_buffers())
        return _IncompatibleKeys(
            missing_keys=list(model_keys - state_dict.keys()),
            unexpected_keys=list(state_dict.keys() - model_keys)
        )
    else:
        return model.load_state_dict(state_dict, strict=strict)


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
        # In sharded models, each shard has only part of the full state_dict, so only gather
        # parameters that are in the current state_dict.
        named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
        params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
        if len(params_to_gather) > 0:
            # Since Zero3 puts placeholders in model params, this context manager gathers the params of
            # the current layer, then loads from the state dict and then re-partitions them again
            with gather_parameter(params_to_gather, modifier_rank=0):
                if get_rank() == 0 or not is_deepspeed_module(module):
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
            strict=strict,
            prefix=f"{prefix}{name}.",
            metadata=metadata,
            error_msgs=error_msgs
        )
