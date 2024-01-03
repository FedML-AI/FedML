import importlib.util

from peft.import_utils import is_bnb_available, is_bnb_4bit_available

try:
    # Starting from transformers v4.33.1, `transformers.deepspeed` is moved to
    #   `transformers.integrations.deepspeed`
    from transformers.integrations.deepspeed import is_deepspeed_available, is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_available, is_deepspeed_zero3_enabled


def _is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def is_fedml_available() -> bool:
    return _fedml_available


def is_flash_attn_available() -> bool:
    import torch.cuda

    return _flash_attn_available and torch.cuda.is_available()


_fedml_available = _is_package_available("fedml")
_flash_attn_available = _is_package_available("flash_attn")
