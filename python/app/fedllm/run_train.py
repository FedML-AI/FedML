from typing import List, Optional, Tuple

from dataclasses import dataclass, field
import warnings

from datasets import Dataset, load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
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
from src.modeling_utils import get_data_collator
from src.trainer_callback import SavePeftModelCallback
from src.typing import ModelType, TokenizerType
from src.utils import save_config, should_process_save


@dataclass
class FinetuningArguments(TrainingArguments):
    task: str = field(default="finetune", metadata={"help": "finetune task type", "choices": FINETUNE_TASKS})

    @property
    def is_instruction_finetune(self) -> bool:
        return self.task == "instruction"


@dataclass
class ModelArguments:
    model_name: str = field(
        default="EleutherAI/pythia-2.8b",
        metadata={
            "help": "model name or checkpoint path.",
            "choices": MODEL_NAMES
        },
    )
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


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "dataset name", "choices": DATASET_NAMES})
    dataset_path: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to the training data file(s).", "nargs": "+"}
    )
    test_dataset_size: int = field(
        default=-1,
        metadata={"help": "test set size. Will be ignored if `dataset_path` has more than 1 entry."},
    )
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "max sequence length."})

    def __post_init__(self) -> None:
        if self.dataset_name is None and len(self.dataset_path) <= 0:
            raise ValueError("\"dataset_name\" and \"dataset_path\" cannot both be empty.")

        if len(self.dataset_path) == 1 and self.test_dataset_size <= 0:
            raise ValueError("\"test_dataset_size\" must be a positive value when dataset_path has only 1 entry.")


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
        dataset_args: DataArguments,
        dataset: Dataset,
        tokenizer: TokenizerType
) -> Dataset:
    remove_columns = list({"text", *dataset.column_names})
    if "text" not in dataset.column_names:
        dataset = dataset.map(_add_text)

    print(f"preprocessing dataset")
    dataset = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            max_length=dataset_args.max_seq_length,
            truncation=True,
            return_overflowing_tokens=True
        ),
        batched=True,
        remove_columns=remove_columns
    )

    print(f"dataset has {dataset.num_rows:,} rows")

    return dataset


def get_dataset(
        dataset_args: DataArguments,
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

        print("splitting dataset")
        dataset_dict = dataset.train_test_split(
            test_size=dataset_args.test_dataset_size,
            shuffle=True,
            seed=seed
        )

        print(f"done preprocessing")
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]
    else:
        train_dataset = preprocess_dataset(dataset_args, dataset_dict["train"], tokenizer)
        test_dataset = preprocess_dataset(dataset_args, dataset_dict["test"], tokenizer)
        print(f"done preprocessing")

    print(f"Train data size: {train_dataset.num_rows:,}")
    print(f"Test data size: {test_dataset.num_rows:,}")
    return train_dataset, test_dataset


def get_tokenizer(model_name: str, add_special_tokens: bool = False) -> TokenizerType:
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO: support additional tokens
    if add_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

    return tokenizer


def get_model(model_args: ModelArguments, tokenizer_length: Optional[int] = None, **kwargs) -> ModelType:
    kwargs.setdefault("trust_remote_code", True)
    kwargs.setdefault("low_cpu_mem_usage", not is_deepspeed_zero3_enabled())

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, **kwargs)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if tokenizer_length is not None and tokenizer_length > embedding_size:
        print(f"Resize embedding to tokenizer length: {tokenizer_length:,}")
        model.resize_token_embeddings(tokenizer_length)

        # update model configurations
        if hasattr(model.config, "vocab_size"):
            model.config.vocab_size = tokenizer_length

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


def get_max_seq_length(model: ModelType, default_max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH) -> int:
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        embedding_size = getattr(model.config, length_setting, None)
        if embedding_size is not None:
            print(f"Found max length: {embedding_size}")
            break
    else:
        embedding_size = default_max_seq_length
        print(f"Using default max length: {embedding_size}")

    return embedding_size


def train() -> None:
    # configs
    parser = HfArgumentParser((ModelArguments, DataArguments, FinetuningArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    # prepare models
    print(f"Loading tokenizer for \"{model_args.model_name}\"")
    tokenizer = get_tokenizer(model_args.model_name, add_special_tokens=training_args.is_instruction_finetune)

    print(f"Loading model for \"{model_args.model_name}\"")
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

        print("Training")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        print(f"Saving model to \"{training_args.output_dir}\"")
        trainer.save_checkpoint(training_args.output_dir)

    if training_args.do_eval:
        print("Evaluating")
        print(trainer.evaluate())


if __name__ == '__main__':
    train()
