from typing import Any, Optional, Sequence, Type, Union

from torch import FloatTensor, LongTensor
from transformers import AutoConfig, AutoModelForCausalLM, StoppingCriteria
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _get_model_class

from .typing import is_model_config_type, is_model_type, ModelConfigType, ModelType, TokenizerType
from .utils import is_directory


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


# Adapted from https://github.com/huggingface/transformers/blob/91d7df58b6537d385e90578dac40204cb550f706/src/transformers/models/auto/auto_factory.py#L407
def get_model_class_from_config(
        config: ModelConfigType,
        cls: _BaseAutoModelClass = AutoModelForCausalLM,
        **kwargs: Any
) -> Type[ModelType]:
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
    has_local_code = type(config) in cls._model_mapping.keys()
    trust_remote_code = resolve_trust_remote_code(
        trust_remote_code, config._name_or_path, has_local_code, has_remote_code
    )

    if has_remote_code and trust_remote_code:
        class_ref = config.auto_map[cls.__name__]
        if "--" in class_ref:
            repo_id, class_ref = class_ref.split("--")
        else:
            repo_id = config.name_or_path
        model_class: Any = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
        if is_directory(config._name_or_path):
            model_class.register_for_auto_class(cls.__name__)
        else:
            cls.register(config.__class__, model_class, exist_ok=True)
        _ = kwargs.pop("code_revision", None)
        return model_class

    elif type(config) in cls._model_mapping.keys():
        return _get_model_class(config, cls._model_mapping)

    else:
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}."
        )


class StopStringCriteria(StoppingCriteria):
    def __init__(
            self,
            tokenizer: TokenizerType,
            stop: Union[str, Sequence[str]],
            num_prompt_tokens: int = 0
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_prompt_tokens = num_prompt_tokens

        if not bool(stop):
            # if empty string or None
            stop = []
        elif isinstance(stop, str):
            stop = [stop]
        self.stop = tuple(s for s in stop if len(s) > 0)

    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        for input_id_tensor in input_ids:
            token_ids = input_id_tensor.tolist()
            delta_token_ids = token_ids[self.num_prompt_tokens:]

            # text = self.tokenizer.decode(token_ids)
            # prompt_text = self.tokenizer.decode(token_ids[:self.num_prompt_tokens])
            delta_text = self.tokenizer.decode(delta_token_ids)
            if delta_text.endswith(self.stop):
                return True

        return False
