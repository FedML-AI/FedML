from typing import Optional

from functools import partial

from peft import PeftModel
import torch
from torch import Tensor
from torch.nn import Module
from transformers import (
    GPTNeoXForCausalLM,
    GPTNeoXModel,
)
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

from ..integrations import is_flash_attn_available

if is_flash_attn_available():
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

SUPPORTED_MODELS = (
    GPTNeoXModel,
    GPTNeoXForCausalLM,
)
MODEL_ATTENTION_MAPPING = {
    GPTNeoXModel: "attention",
}


def add_flash_attention(model: Module) -> None:
    """
    Helper function for patching HF language models with Flash Attention.

    Limitations:
      - Flash attention requires CUDA and fp16/bf16 training. It also requires contiguous attention masks.

    Args:
        model: model to patch.

    Returns:
        None

    """
    if not is_flash_attn_available():
        raise ModuleNotFoundError(
            "Please install FlashAttention with `pip install flash-attn --no-build-isolation`. See"
            " https://github.com/Dao-AILab/flash-attention#installation-and-features for detail."
        )

    if isinstance(model, PeftModel):
        # only patch the base model
        model = model.get_base_model()

    if not isinstance(model, SUPPORTED_MODELS):
        raise TypeError(
            f"Model patching does not support `{model.__class__.__name__}`."
        )

    if isinstance(model, GPTNeoXForCausalLM):
        model = model.gpt_neox

    layers = model.layers
    attention_key = MODEL_ATTENTION_MAPPING.get(model.__class__, "attention")
    for i, layer in enumerate(layers):
        _add_flash_attention(getattr(layer, attention_key))


# Adapted from https://github.com/LAION-AI/Open-Assistant/blob/c2c1318694e35566473c69d4a9cc374a86cff12c/model/model_training/models/patching.py#L63
def _add_flash_attention(module: Module) -> None:
    """
    Replaces the standard attention implementation with Flash Attention.

    Limitations:
      - Only works for fp16 or bf16 inputs.
      - Requires inputs to be on CUDA.
      - `output_attentions=True` does not work after patching, attention weights will be None.
      - Non-contiguous attention masks are not supported (e.g. [1, 1, 0, 1, 1, 0, 0] will
        just become [1, 1, 1, 1, 1, 0, 0]).

    Args:
        module: Attention Module to patch.

    Returns:
        None

    """
    if isinstance(module, GPTNeoXAttention):
        from .modeling_gpt_neox import gpt_neox_flash_attention

        if not hasattr(module, "_attn"):
            raise AttributeError(
                f"Provided {module.__class__.__name__} module doesn't have a _attn() function to be patched."
            )
        module._attn = partial(gpt_neox_flash_attention, module)
    else:
        raise NotImplementedError(f"Flash attention is not implemented for {module.__class__.__name__}.")


# Adapted from https://github.com/LAION-AI/Open-Assistant/blob/1e6e56975b12a8d6ecc3253ef4e42e496136c7a0/model/model_training/models/patching_utils.py#L5
# and https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/b89ab5ae701b2f60731aceeaece00408b0b7bbe7/training/utils/llama_patch.py#L28
def flash_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: Optional[bool] = False,
        training: bool = False
) -> Tensor:
    if head_mask is not None:
        raise ValueError("head_mask different than None is unsupported.")

    attn_func_kwargs = dict(
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        training=training,
    )

    # q, k, v: [batch_size, seq_len, num_attention_heads, attn_head_size]
    # attention_mask (float): [batch_size, seq_len]
    batch_size, seq_len = q.size(0), q.size(1)

    # for gpt-neo-x and gpt-j the query and keys are always in fp32 and the output has the same dtype
    # as value, thus we need to cast them before and after flash attention
    out_dtype = attn_dtype = v.dtype
    if attn_dtype not in (torch.float16, torch.bfloat16):
        # flash attention only support fp16/bf16
        attn_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # pack q, k, v
    qkv = torch.stack([q, k, v], dim=2).to(attn_dtype)

    if attention_mask is None:
        out = flash_self_attention_qkvpacked(qkv, **attn_func_kwargs)
    else:
        # Limitation: non-contiguous attention mask will not be handled correctly
        # model will be able to pay attention between the first and last non-masked
        # token, i.e. left- and right-side padding is supported.
        padding_mask = (attention_mask >= 0)
        qkv, indices, cu_seqlens, max_seqlen = unpad_input(qkv, padding_mask)

        # out: [num_unmasked_tokens, num_attention_heads, attn_head_size]
        out = flash_self_attention_qkvpacked(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **attn_func_kwargs)
        out = pad_input(out, indices, batch_size, seq_len)

    out = out.to(out_dtype)
    return out


# Adapted from https://github.com/Dao-AILab/flash-attention/blob/0705d2718dd39a39507dbdac85c538189a8436a1/flash_attn/modules/mha.py#L36
def flash_self_attention_qkvpacked(
        qkv: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: Optional[bool] = False,
        training: bool = False
) -> Tensor:
    assert qkv.dtype in [torch.float16, torch.bfloat16]
    assert qkv.is_cuda

    if not training:
        dropout_p = 0.0

    if cu_seqlens is not None:
        assert cu_seqlens.dtype == torch.int32
        assert isinstance(max_seqlen, int)
        return flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal
        )
    else:
        return flash_attn_qkvpacked_func(
            qkv,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal
        )
