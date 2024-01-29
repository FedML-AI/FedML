import importlib.util

from accelerate.utils import compare_versions


def _is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


_flash_attn_available = _is_package_available("flash_attn")
_jinja2_available = _is_package_available("jinja2")


# Adapted from https://github.com/huggingface/transformers/blob/80377eb018c077dba434bc8e7912bcaed3a64d09/src/transformers/utils/import_utils.py#L617
def is_flash_attn_available() -> bool:
    if not _flash_attn_available:
        return False

    # Let's add an extra check to see if cuda is available
    import torch.cuda

    if not torch.cuda.is_available():
        return False

    if torch.version.cuda:
        return compare_versions("flash_attn", ">=", "2.1.0")
    elif torch.version.hip:
        # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
        return compare_versions("flash_attn", ">=", "2.0.4")
    else:
        return False


def is_jinja2_available() -> bool:
    return (
            _jinja2_available
            and compare_versions("jinja2", ">=", "3.0.0")
    )
