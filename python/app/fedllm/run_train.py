from typing import Any, List, Optional, Tuple, Union

from dataclasses import dataclass, field
import logging
import os
import warnings

from accelerate.utils import compare_versions
from datasets import Dataset, load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.deepspeed import is_deepspeed_zero3_enabled

from src.constants import (
    DATASET_NAMES,
    DEFAULT_MAX_SEQ_LENGTH,
    END_KEY,
    FINETUNE_TASKS,
    INSTRUCTION_KEY,
    MODEL_NAMES,
    PROMPT_NO_INPUT_FORMAT,
    PROMPT_WITH_INPUT_FORMAT,
    RESPONSE_KEY_NL,
)
from src.hf_trainer import HFTrainer
from src.modeling_utils import get_data_collator, get_max_seq_length as _get_max_seq_length, get_vocab_size
from src.trainer_callback import SavePeftModelCallback
from src.typing import ModelConfigType, ModelType, TokenizerType
from src.utils import is_directory, is_file, parse_hf_args, save_config, should_process_save


@dataclass
class FinetuningArguments(TrainingArguments):
    task: str = field(default="finetune", metadata={"help": "finetune task type", "choices": FINETUNE_TASKS})

    @property
    def is_instruction_finetune(self) -> bool:
        return self.task == "instruction"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="EleutherAI/pythia-70m", metadata={"help": "model name or path."})
    use_lora: bool = field(default=False, metadata={"help": "Set to `True` to enable LoRA."})
    lora_r: int = field(default=8, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout."})
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
            "nargs": "+",
        },
    )
    auth_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Authentication token for Hugging Face private models such as Llama 2.",
        },
    )
    load_pretrained: bool = field(default=True, metadata={"help": "If `True`, load pretrained model."})

    def __post_init__(self) -> None:
        if is_file(self.model_name_or_path):
            if self.load_pretrained:
                raise ValueError(
                    "\"model_name_or_path\" must be a model ID or directory path if \"load_pretrained\" is `True`."
                )

        elif not is_directory(self.model_name_or_path):
            # if model_name_or_path is not a local file or directory
            if self.model_name_or_path not in MODEL_NAMES:
                model_names_str = "', '".join(MODEL_NAMES)

                raise ValueError(
                    f"\"model_name_or_path\" must be a valid file/directory path or a supported model ID ("
                    f"choose from '{model_names_str}') but received \"{self.model_name_or_path}\"."
                )

            if self.model_name_or_path.startswith("meta-llama/Llama-2-"):
                if compare_versions("transformers", "<", "4.31.0"):
                    raise NotImplementedError(f"{self.model_name_or_path} requires transformers >= 4.31.0")

                if self.auth_token is not None:
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = str(self.auth_token)

                # need to verify if already logged in
                from huggingface_hub import HfApi
                from huggingface_hub.utils import LocalTokenNotFoundError

                try:
                    HfApi().whoami()
                except LocalTokenNotFoundError:
                    raise LocalTokenNotFoundError(
                        f"Token is required for {self.model_name_or_path}, but no token found. You need to provide a"
                        f" token or be logged in to Hugging Face."
                        f"\nTo pass a token, you could pass `--auth_token \"<your token>\"` or set environment"
                        f" variable `HUGGING_FACE_HUB_TOKEN=\"${{your_token}}\"`."
                        f"\nTo login, use `huggingface-cli login` or `huggingface_hub.login`."
                        f" See https://huggingface.co/settings/tokens."
                    )


@dataclass
class DatasetArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "dataset name", "choices": DATASET_NAMES})
    dataset_path: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to the training data file(s).", "nargs": "+"}
    )
    test_dataset_size: int = field(
        default=-1,
        metadata={"help": "test set size. Will be ignored if `dataset_path` has more than 1 entry."}
    )
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "max sequence length."})
    truncate_long_seq: bool = field(
        default=True,
        metadata={"help": "Whether to truncate long sequences whose length > max_seq_length."}
    )
    remove_long_seq: bool = field(
        default=False,
        metadata={"help": "Whether to remove all data whose token length > max_seq_length."}
    )

    def __post_init__(self) -> None:
        if self.dataset_name is None and len(self.dataset_path) <= 0:
            raise ValueError("\"dataset_name\" and \"dataset_path\" cannot both be empty.")

        if len(self.dataset_path) == 1 and self.test_dataset_size <= 0:
            raise ValueError("\"test_dataset_size\" must be a positive value when dataset_path has only 1 entry.")

        if self.remove_long_seq and not self.truncate_long_seq:
            warnings.warn("\"truncate_long_seq\" is set to `True` since \"remove_long_seq\" is `True`.")
            self.truncate_long_seq = True

    @property
    def truncation_max_length(self) -> Optional[int]:
        if self.max_seq_length is not None and self.remove_long_seq:
            # set to max_seq_length + 1 so that sequences has length >= max_seq_lengths can be filtered
            return self.max_seq_length + 1
        else:
            return self.max_seq_length


def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    context = rec.get("context")

    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")

    if not response:
        raise ValueError(f"Expected a response in: {rec}")

    # For some instructions there is an input that goes along with the instruction, providing context for the
    # instruction. For example, the input might be a passage from Wikipedia and the instruction says to extract
    # some piece of information from it. The response is that information to extract. In other cases there is
    # no input. For example, the instruction might be open QA such as asking what year some historic figure was
    # born.
    if context:
        rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
    else:
        rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
    return rec


def preprocess_dataset(
        dataset_args: DatasetArguments,
        dataset: Dataset,
        tokenizer: TokenizerType
) -> Dataset:
    remove_columns = list({"text", *dataset.column_names})
    if "text" not in dataset.column_names:
        dataset = dataset.map(_add_text)

    tokenization_kwargs = dict(
        truncation=dataset_args.truncate_long_seq,
        max_length=dataset_args.truncation_max_length,
    )

    logging.info(f"preprocessing dataset")
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], **tokenization_kwargs),
        batched=True,
        remove_columns=remove_columns
    )

    if dataset_args.remove_long_seq and dataset_args.max_seq_length is not None:
        dataset = dataset.filter(lambda rec: len(rec["input_ids"]) <= dataset_args.max_seq_length)
        logging.info(f"dataset has {dataset.num_rows:,} rows after filtering for truncated records")

    logging.info(f"dataset has {dataset.num_rows:,} rows")

    return dataset


def get_dataset(
        dataset_args: DatasetArguments,
        tokenizer: TokenizerType,
        seed: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    if len(dataset_args.dataset_path) >= 2:
        warnings.warn("More than 2 dataset paths provided. Only the first 2 will be loaded.")
        data_files = {"train": dataset_args.dataset_path[0], "test": dataset_args.dataset_path[1]}
    elif len(dataset_args.dataset_path) == 0:
        if dataset_args.dataset_name is None:
            raise ValueError("\"dataset_name\" and \"dataset_path\" cannot both be empty.")

        data_files = None
    else:
        data_files = dataset_args.dataset_path

    if dataset_args.dataset_name is not None:
        dataset_dict = load_dataset(dataset_args.dataset_name, data_files=data_files)
    else:
        dataset_dict = load_dataset("json", data_files=data_files)

    if len(dataset_dict.keys()) == 1:
        dataset = preprocess_dataset(dataset_args, dataset_dict["train"], tokenizer)

        logging.info("splitting dataset")
        dataset_dict = dataset.train_test_split(
            test_size=dataset_args.test_dataset_size,
            shuffle=True,
            seed=seed
        )

        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]
    else:
        train_dataset = preprocess_dataset(dataset_args, dataset_dict["train"], tokenizer)
        test_dataset = preprocess_dataset(dataset_args, dataset_dict["test"], tokenizer)

    logging.info(f"done preprocessing")

    logging.info(f"Train data size: {train_dataset.num_rows:,}")
    logging.info(f"Test data size: {test_dataset.num_rows:,}")
    return train_dataset, test_dataset


def get_tokenizer(model_args: ModelArguments, add_special_tokens: bool = False, **kwargs) -> TokenizerType:
    kwargs.setdefault("trust_remote_code", True)

    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO: support additional tokens
    if add_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

    return tokenizer


def get_model(model_args: ModelArguments, tokenizer_length: Optional[int] = None, **kwargs) -> ModelType:
    kwargs.setdefault("trust_remote_code", True)

    if model_args.load_pretrained:
        kwargs.setdefault("low_cpu_mem_usage", not is_deepspeed_zero3_enabled())

        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **kwargs)
    else:
        # see https://discuss.huggingface.co/t/how-to-load-model-without-pretrained-weight/34155/3
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **kwargs)
        model = AutoModelForCausalLM.from_config(config)

    model_vocab_size = get_vocab_size(model.config)
    if tokenizer_length is not None and model_vocab_size < tokenizer_length:
        logging.info(f"Resize embedding from {model_vocab_size:,} to tokenizer length: {tokenizer_length:,}")
        model.resize_token_embeddings(tokenizer_length)

        # model.config should also be updated
        assert model.config.vocab_size == tokenizer_length

    if model_args.use_lora:
        # apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            target_modules=model_args.lora_target_modules,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout
        )
        # see https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

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


def train() -> None:
    # configs
    model_args, dataset_args, training_args = parse_hf_args((ModelArguments, DatasetArguments, FinetuningArguments))

    # prepare models
    logging.info(f"Loading tokenizer for \"{model_args.model_name_or_path}\"")
    tokenizer = get_tokenizer(model_args, add_special_tokens=training_args.is_instruction_finetune)

    logging.info(f"Loading model for \"{model_args.model_name_or_path}\"")
    model = get_model(model_args, tokenizer_length=len(tokenizer), use_cache=not training_args.gradient_checkpointing)

    if dataset_args.max_seq_length is None:
        dataset_args.max_seq_length = get_max_seq_length(model)

    # dataset
    with training_args.main_process_first():
        train_dataset, test_dataset = get_dataset(
            dataset_args=dataset_args,
            tokenizer=tokenizer,
            seed=training_args.seed
        )

    trainer = HFTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=get_data_collator(
            tokenizer,
            escape_token=RESPONSE_KEY_NL if training_args.is_instruction_finetune else None,
            pad_to_multiple_of=dataset_args.max_seq_length
        ),
        callbacks=[
            # save peft adapted model weights
            SavePeftModelCallback(),
        ]
    )

    if training_args.do_train:
        if should_process_save(trainer):
            # save model config before training
            save_config(model, training_args.output_dir)

        logging.info("Training")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        logging.info(f"Saving model to \"{training_args.output_dir}\"")
        trainer.save_checkpoint(training_args.output_dir)

    if training_args.do_eval:
        logging.info("Evaluating")
        logging.info(trainer.evaluate())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train()
