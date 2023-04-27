from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset, load_dataset
import numpy as np
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTJModel,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

MODEL_NAMES = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-j-6B",
]
DEFAULT_MAX_SEQ_LENGTH = 1024

INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
PROMPT_NO_INPUT_FORMAT = f"""{INTRO_BLURB}

{INSTRUCTION_KEY}
{{instruction}}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""

PROMPT_WITH_INPUT_FORMAT = f"""{INTRO_BLURB}

{INSTRUCTION_KEY}
{{instruction}}

{INPUT_KEY}
{{input}}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""

ModelType = Union[GPTJModel, GPTNeoXForCausalLM, PeftModelForCausalLM]
TokenizerType = Union[GPTNeoXTokenizerFast]


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
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
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
    dataset_path: List[str] = field(metadata={"help": "Path to the training data file(s).", "nargs": "+"})
    test_dataset_size: int = field(
        default=1_000,
        metadata={"help": "test set size. Will be ignored if `dataset_path` has more than one entry."},
    )
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "max sequence length."})


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ):
        # see https://github.com/huggingface/peft/issues/96#issuecomment-1460080427
        checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        model = kwargs.get("model", None)

        if isinstance(model, PeftModel):
            peft_model_path = checkpoint_dir / "adapter_model"
            model.save_pretrained(str(peft_model_path))

        return control


def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    context = rec.get("context")

    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")

    if not response:
        raise ValueError(f"Expected a response in: {rec}")

    # For some instructions there is an input that goes along with the instruction, providing context for the
    # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
    # some piece of information from it.  The response is that information to extract.  In other cases there is
    # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
    # born.
    if context:
        rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
    else:
        rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
    return rec


def preprocess_dataset(
        dataset: Dataset,
        tokenizer: TokenizerType,
        max_length: int
) -> Dataset:
    dataset = dataset.map(_add_text)

    print(f"preprocessing dataset")
    dataset = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        ),
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"]
    )

    print(f"dataset has {dataset.num_rows:,} rows")
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    print(f"dataset has {dataset.num_rows:,} rows after filtering for truncated records")

    return dataset


def get_dataset(
        dataset_path: Union[str, Sequence[str]],
        tokenizer: TokenizerType,
        max_length: int,
        seed: Optional[int] = None,
        test_dataset_size: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    if isinstance(dataset_path, str):
        dataset_path = [dataset_path]

    assert len(dataset_path) > 0, "Received empty dataset_path"
    print(f"loading the following datasets:\n\t")
    print(f"\n\t".join(dataset_path))

    # TODO: cleanup
    if len(dataset_path) == 1:
        assert test_dataset_size is not None, f"test_dataset_size is required when len(dataset_path) == 1"
    elif len(dataset_path) >= 2:
        dataset_path = {"train": dataset_path[0], "test": dataset_path[1]}

    dataset_dict = load_dataset("json", data_files=dataset_path)
    if len(dataset_dict.keys()) == 1:
        dataset = preprocess_dataset(dataset_dict["train"], tokenizer, max_length)

        print("shuffling dataset")
        dataset = dataset.shuffle(seed)

        print("splitting dataset")
        dataset_dict = dataset.train_test_split(test_size=test_dataset_size, seed=seed)

        print(f"done preprocessing")
        train_dataset = dataset_dict["train"]
        test_dataset = dataset_dict["test"]
    else:
        train_dataset = preprocess_dataset(dataset_dict["train"], tokenizer, max_length)
        test_dataset = preprocess_dataset(dataset_dict["test"], tokenizer, max_length)
        print(f"done preprocessing")

    print(f"Train data size: {train_dataset.num_rows:,}")
    print(f"Test data size: {test_dataset.num_rows:,}")
    return train_dataset, test_dataset


def get_tokenizer(model_name: str) -> TokenizerType:
    tokenizer: TokenizerType = AutoTokenizer.from_pretrained(model_name)

    # TODO: scrutinize
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

    return tokenizer


def get_model(model_args: ModelArguments, tokenizer_length: Optional[int] = None, **kwargs) -> ModelType:
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, trust_remote_code=True, **kwargs)

    print(f"Resize embedding to tokenizer length: {tokenizer_length:,}")
    # TODO: resize when tokenizer_length < model embedding size?
    model.resize_token_embeddings(tokenizer_length)

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


def get_data_collator(tokenizer: TokenizerType) -> DataCollatorForCompletionOnlyLM:
    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8
    )


def train() -> None:
    # configs
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    # prepare models
    print(f"Loading tokenizer for \"{model_args.model_name}\"")
    tokenizer = get_tokenizer(model_args.model_name)

    print(f"Loading model for \"{model_args.model_name}\"")
    model = get_model(model_args, tokenizer_length=len(tokenizer), use_cache=not training_args.gradient_checkpointing)

    if dataset_args.max_seq_length is None:
        dataset_args.max_seq_length = get_max_seq_length(model)

    # dataset
    train_dataset, test_dataset = get_dataset(
        dataset_path=dataset_args.dataset_path,
        tokenizer=tokenizer,
        max_length=dataset_args.max_seq_length,
        seed=training_args.seed,
        test_dataset_size=dataset_args.test_dataset_size
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=get_data_collator(tokenizer),
        callbacks=[
            # save peft adapted model weights
            SavePeftModelCallback,
        ]
    )

    print("Training")
    trainer.train()

    print(f"Saving model to \"{training_args.output_dir}\"")
    trainer.save_state()
    trainer.save_model()


if __name__ == '__main__':
    train()
