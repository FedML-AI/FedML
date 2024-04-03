from typing import Any, Optional, Sized, Tuple, Union

import logging

from datasets import IterableDataset, IterableDatasetDict, load_dataset
from peft import get_peft_model
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .configurations import DatasetArguments, ModelArguments
from .constants import DEFAULT_MAX_SEQ_LENGTH
from .dataset_utils import get_keyword_replacer, get_prompt_formatter
from .integrations import is_deepspeed_zero3_enabled
from .modeling_utils import (
    get_max_seq_length as _get_max_seq_length,
    get_model_class_from_config,
    get_parameter_stats_repr,
    get_vocab_size,
)
from .models import add_flash_attention
from .typing import DatasetType, ModelConfigType, ModelType, TokenizerType
from .utils import replace_if_exists


def preprocess_dataset(
        dataset_args: DatasetArguments,
        dataset: DatasetType,
        tokenizer: TokenizerType
) -> DatasetType:
    dataset_kwargs = {}
    if not isinstance(dataset, IterableDataset):
        dataset_kwargs["num_proc"] = dataset_args.dataset_num_proc
        dataset_kwargs["desc"] = None

    if not dataset_args.column_name_mapping.keys().isdisjoint(dataset.column_names):
        # replace overlapping columns
        column_name_mapping = {
            k: dataset_args.column_name_mapping[k]
            for k in set(dataset_args.column_name_mapping.keys()).intersection(dataset.column_names)
        }

        dataset = dataset.rename_columns(column_name_mapping)

    columns_to_remove = {"text", *dataset.column_names}
    if "text" not in dataset.column_names:
        dataset = dataset.map(
            get_prompt_formatter(dataset_args.prompt_style),
            **replace_if_exists(dataset_kwargs, desc="Apply text template")
        )
    if not dataset_args.disable_data_keyword_replacement:
        dataset = dataset.map(
            get_keyword_replacer(dataset_args.data_keyword_replacements),
            **replace_if_exists(dataset_kwargs, desc="Replace keywords")
        )

    tokenization_kwargs = dict(
        truncation=dataset_args.truncate_long_seq,
        max_length=dataset_args.truncation_max_length,
    )

    def encode(batch):
        return tokenizer(batch["text"], **tokenization_kwargs)

    if dataset_args.tokenize_on_the_fly:
        if dataset_args.remove_long_seq:
            raise ValueError("`remove_long_seq` is not compatible with `tokenize_on_the_fly`")

        dataset = dataset.remove_columns(list(columns_to_remove - {"text"}))
        dataset.set_transform(encode)

    else:
        logging.info(f"preprocessing dataset")
        dataset = dataset.map(
            encode,
            batched=True,
            remove_columns=list(columns_to_remove),
            **replace_if_exists(dataset_kwargs, desc="Tokenize")
        )

    if isinstance(dataset, Sized):
        logging.info(f"dataset has {len(dataset):,} rows")

    if dataset_args.remove_long_seq and dataset_args.max_seq_length is not None:
        dataset = dataset.filter(
            lambda rec: len(rec["input_ids"]) <= dataset_args.max_seq_length,
            **replace_if_exists(dataset_kwargs, desc="Remove long sequence")
        )

        if isinstance(dataset, Sized):
            logging.info(f"dataset has {len(dataset):,} rows after filtering for truncated records")

    return dataset


def get_dataset(
        dataset_args: DatasetArguments,
        tokenizer: TokenizerType,
        seed: Optional[int] = None,
        is_local_main_process: bool = True
) -> Tuple[DatasetType, DatasetType, DatasetType]:
    dataset_kwargs = dict(
        path="json",
        name=dataset_args.dataset_config_name,
        streaming=dataset_args.dataset_streaming,
    )

    if dataset_args.dataset_name is not None:
        dataset_kwargs["path"] = dataset_args.dataset_name
        dataset_kwargs["data_files"] = None

    elif len(dataset_args.dataset_path) >= 2:
        dataset_kwargs["data_files"] = {
            "train": dataset_args.dataset_path[0],
            "test": dataset_args.dataset_path[1],
        }

    elif len(dataset_args.dataset_path) == 0:
        raise ValueError("\"dataset_name\" and \"dataset_path\" cannot both be empty.")

    else:
        dataset_kwargs["data_files"] = dataset_args.dataset_path

    dataset_dict = load_dataset(**dataset_kwargs)
    if (
            dataset_args.cleanup_data_cache and
            is_local_main_process and
            not isinstance(dataset_dict, IterableDatasetDict)
    ):
        # only cleanup cache on local main process (i.e. local_rank == 0)
        dataset_dict.cleanup_cache_files()

    if len(dataset_dict.keys()) == 1:
        if dataset_args.test_size is None:
            raise ValueError(
                "The dataset only has 1 split. A positive `test_dataset_ratio` or `test_dataset_size`"
                " is required."
            )

        dataset = preprocess_dataset(dataset_args, dataset_dict["train"], tokenizer)

        logging.info("splitting dataset")
        if isinstance(dataset, IterableDataset):
            dataset = dataset.shuffle(seed)

            train_dataset = dataset.skip(dataset_args.test_size)
            test_dataset = dataset.take(dataset_args.test_size)
        else:
            dataset_dict = dataset.train_test_split(
                test_size=dataset_args.test_size,
                shuffle=True,
                seed=seed
            )

            train_dataset = dataset_dict["train"]
            test_dataset = dataset_dict["test"]
    else:
        train_dataset = preprocess_dataset(dataset_args, dataset_dict["train"], tokenizer)
        test_dataset = preprocess_dataset(dataset_args, dataset_dict["test"], tokenizer)

    logging.info(f"done preprocessing")

    if dataset_args.eval_dataset_size <= 0:
        eval_dataset = test_dataset
    elif isinstance(test_dataset, IterableDataset):
        eval_dataset = test_dataset.take(dataset_args.eval_dataset_size)
    else:
        eval_dataset = test_dataset.select(range(min(len(test_dataset), dataset_args.eval_dataset_size)))

    if isinstance(train_dataset, Sized):
        logging.info(f"Train dataset size: {len(train_dataset):,}")
    if isinstance(eval_dataset, Sized):
        logging.info(f"Eval dataset size: {len(eval_dataset):,}")
    if isinstance(test_dataset, Sized):
        logging.info(f"Test dataset size: {len(test_dataset):,}")
    return train_dataset, test_dataset, eval_dataset


def get_tokenizer(model_args: ModelArguments, **kwargs: Any) -> TokenizerType:
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(**model_args.get_tokenizer_kwargs(**kwargs))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_base_model(
        model_args: ModelArguments,
        tokenizer_length: Optional[int] = None,
        **kwargs: Any
) -> ModelType:
    kwargs = model_args.get_model_kwargs(**kwargs)
    config: Optional[ModelConfigType] = kwargs.pop("config", None)

    if model_args.load_pretrained:
        kwargs.setdefault("low_cpu_mem_usage", not is_deepspeed_zero3_enabled())

        model: ModelType = AutoModelForCausalLM.from_pretrained(**kwargs)
    else:
        if config is None:
            config: ModelConfigType = AutoConfig.from_pretrained(**kwargs)
        torch_dtype: Optional[torch.dtype] = kwargs.get("torch_dtype", model_args.torch_dtype)

        # see https://discuss.huggingface.co/t/how-to-load-model-without-pretrained-weight/34155/3
        model: ModelType = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)

    model_vocab_size = get_vocab_size(model.config)
    if tokenizer_length is not None and model_vocab_size < tokenizer_length:
        logging.info(f"Resize embedding from {model_vocab_size:,} to tokenizer length: {tokenizer_length:,}")
        model.resize_token_embeddings(tokenizer_length)

        # model.config should also be updated
        assert model.config.vocab_size == tokenizer_length

    if model_args.use_flash_attention and not getattr(model, "_supports_flash_attn_2", False):
        # patch models that do not support `use_flash_attention_2`
        add_flash_attention(model)

    return model


def get_model(model_args: ModelArguments, tokenizer_length: Optional[int] = None, **kwargs) -> ModelType:
    kwargs = model_args.get_model_kwargs(**kwargs)
    torch_dtype = kwargs.get("torch_dtype", model_args.torch_dtype)

    model = get_base_model(model_args, tokenizer_length, **kwargs)
    peft_config = model_args.get_peft_config(
        model,
        revision=kwargs.get("revision", model_args.model_revision)
    )

    if peft_config is not None:
        # see https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

        # TODO: support non-LoRA module saving when `lora_on_all_modules=True`
        # if model_args.peft_type == "lora" and model_args.lora_on_all_modules:
        #     from peft.tuners.lora import LoraLayer
        #
        #     # enable gradient for non-LoRA layers
        #     lora_layer_prefixes = tuple({n for n, m in model.named_modules() if isinstance(m, LoraLayer)})
        #
        #     for n, p in model.named_parameters():
        #         if not n.startswith(lora_layer_prefixes):
        #             p.requires_grad = True

    if torch_dtype is not None:
        # convert PEFT weights to `torch_dtype`
        logging.info(f"Loading model in {torch_dtype}.")
        model.to(torch_dtype)
        model.config.torch_dtype = torch_dtype

    # print model parameter stats
    print(get_parameter_stats_repr(model), flush=True)

    return model


def get_max_seq_length(
        model_or_config: Union[str, ModelConfigType, ModelType],
        default_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        **kwargs: Any
) -> int:
    embedding_size = _get_max_seq_length(model_or_config, **kwargs)

    if embedding_size is not None:
        logging.info(f"Found max length: {embedding_size}")
    else:
        embedding_size = default_max_seq_length
        logging.info(f"Using default max length: {embedding_size}")

    return embedding_size
