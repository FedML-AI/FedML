from typing import Any, Dict, Optional

from functools import partial
import re

import evaluate
import numpy as np
import torch
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from train import (
    DataArguments,
    get_data_collator,
    get_dataset,
    get_max_seq_length,
    get_model,
    get_tokenizer,
    IGNORE_INDEX,
    ModelArguments,
    SavePeftModelCallback,
)


def answer_extraction(response: str, answer_type: Optional[str] = None) -> str:
    """
    Adapted from https://github.com/OptimalScale/LMFlow/blob/main/src/lmflow/utils/data_utils.py

    Use this function to extract answers from generated text

    Args:
        response: model generated text
        answer_type: answer type

    Returns:
        Decoded answer (such as A, B, C, D, E for mutiple-choice QA).

    """

    # temp = response["generated_text"]
    temp = response
    if answer_type in ("pubmedqa", "pubmed"):
        # pattern = "Output: (yes|no|maybe)"
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:] if sttr is not None else "N/A"
        pattern = r"(answer|Answer|ANSWER|output|Output|OUTPUT|A): \(*(yes|Yes|YES|no|No|NO|maybe|Maybe|MAYBE)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            mid_answer = mid_answer.split(":")[-1].strip()
            answer = mid_answer.lower()
        else:
            pattern = r"(yes|Yes|YES|no|No|NO|maybe|Maybe|MAYBE)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                answer = sttr.group(0)[:-1].lower()
            else:
                answer = "N/A"
        return answer

    elif answer_type == "medmcqa":
        # pattern = "Output: (A|B|C|D)."
        pattern = "(answer|Answer|ANSWER|output|Output|OUTPUT|A): \(*(A|B|C|D|a|b|c|d)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            answer = mid_answer[-1].lower()
        else:
            pattern = "\(*(A|B|C|D|a|b|c|d)\)*(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                if '(' in sttr.group(0):
                    answer = sttr.group(0)[1].lower()
                else:
                    answer = sttr.group(0)[0].lower()
            else:
                answer = "N/A"
        return answer

    elif answer_type in ("usmle", "medqa-usmle"):
        # pattern = "Output: (A|B|C|D)."
        # sttr = re.search(pattern, temp)
        # answer = sttr.group(0)[8:-1].lower() if sttr is not None else "N/A"
        pattern = "(Answer|Output|A): \(*(A|B|C|D|a|b|c|d)"
        sttr = re.search(pattern, temp)
        if sttr is not None:
            mid_answer = sttr.group(0)
            answer = mid_answer[-1].lower()
        else:
            pattern = "\(*(A|B|C|D|a|b|c|d)\)*(\.|\s)"
            sttr = re.search(pattern, temp)
            if sttr is not None:
                if '(' in sttr.group(0):
                    answer = sttr.group(0)[1].lower()
                else:
                    answer = sttr.group(0)[0].lower()
            else:
                answer = "N/A"
        return answer
    elif answer_type == "text":
        return response
    else:
        raise NotImplementedError(f"Unsupported answer type: {answer_type}")


def compute_acc(
        eval_pred: EvalPrediction,
        tokenizer: PreTrainedTokenizer,
        ignore_idx: int = IGNORE_INDEX,
        answer_type: str = "text"
) -> Dict[str, Any]:
    metric_func = evaluate.load("accuracy")

    label_strs = []
    pred_strs = []

    for i in range(len(eval_pred.label_ids)):
        labels = eval_pred.label_ids[i]
        preds = eval_pred.predictions[i]

        mask = labels != ignore_idx
        indices: np.ndarray = np.nonzero(mask)[0]
        start_idx = int(indices.min())

        label_str = tokenizer.decode(labels[mask], skip_special_tokens=True)
        label_str = label_str.strip()
        pred_str = tokenizer.decode(preds[start_idx:], skip_special_tokens=True)
        pred_str = answer_extraction(pred_str, answer_type)

        label_strs.append(label_str)
        pred_strs.append(pred_str)

        metric_func.add(predictions=int(pred_str.lower() == label_str.lower()), references=1)

    # metric_func.add_batch(predictions=preds, references=labels)

    res = metric_func.compute()
    return res


def main() -> None:
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

    def _preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
        # use this function to solve OOM
        # see https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
        return torch.argmax(logits, dim=-1)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=get_data_collator(tokenizer, pad_to_multiple_of=dataset_args.max_seq_length),
        callbacks=[
            # save peft adapted model weights
            SavePeftModelCallback,
        ],
        compute_metrics=partial(compute_acc, tokenizer=tokenizer, answer_type="pubmedqa"),
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics
    )
    print(trainer.evaluate())


if __name__ == '__main__':
    main()
