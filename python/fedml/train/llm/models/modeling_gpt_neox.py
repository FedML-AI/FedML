from typing import Optional, Tuple

from torch import Tensor
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

from .attention import flash_attention


# Adapted from https://github.com/LAION-AI/Open-Assistant/blob/4f02095c511620ee42f4a0eac67011c5e09d4182/model/model_training/models/patching_neox.py#L8
# and https://github.com/huggingface/optimum/blob/659cf0267706e48c711e7c503d3057b7538efe0e/optimum/bettertransformer/models/decoder_models.py#L137
def gpt_neox_flash_attention(
        self: GPTNeoXAttention,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    # query, key, value: [batch_size, num_attention_heads, seq_len, attn_head_size]
    out_dtype = value.dtype
    q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    if attention_mask is not None:
        attention_mask = attention_mask[:, 0, 0, :]
    out = flash_attention(
        q,
        k,
        v,
        attention_mask,
        head_mask,
        dropout_p=self.attention_dropout.p,
        causal=True,
        training=self.training
    )
    out = out.transpose(1, 2).to(out_dtype)
    return out, None
