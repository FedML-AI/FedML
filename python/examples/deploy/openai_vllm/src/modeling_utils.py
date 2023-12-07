from typing import Any, Optional, Union

from transformers import AutoConfig

from .typing import is_model_config_type, is_model_type, ModelConfigType, ModelType


def get_max_seq_length(model_or_config: Union[str, ModelConfigType, ModelType], **kwargs: Any) -> Optional[int]:
    if is_model_config_type(model_or_config):
        config = model_or_config
    elif is_model_type(model_or_config):
        config = model_or_config.config
    elif isinstance(model_or_config, str):
        config = AutoConfig.from_pretrained(model_or_config, **kwargs)
    else:
        raise TypeError(f"\"{type(model_or_config)}\" is not a supported model_or_config type.")

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        embedding_size = getattr(config, length_setting, None)
        if embedding_size is not None:
            return embedding_size
    else:
        return None
