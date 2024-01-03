from torch import nn

from .integrations import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb

LORA_LAYER_TYPES = [
    nn.Conv2d,
    nn.Embedding,
    nn.Linear,
]
if is_bnb_available():
    LORA_LAYER_TYPES.append(bnb.nn.Linear8bitLt)

    if is_bnb_4bit_available():
        LORA_LAYER_TYPES.append(bnb.nn.Linear4bit)

LORA_LAYER_TYPES = tuple(LORA_LAYER_TYPES)
