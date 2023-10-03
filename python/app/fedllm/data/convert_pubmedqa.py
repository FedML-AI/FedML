from typing import List, Union

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path

from datasets import (
    concatenate_datasets,
    get_dataset_split_names,
    load_dataset,
)

OUTPUT_COLUMNS = {
    "instruction",
    "context",
    "response",
    "category",
}


def _get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=".data/PubMedQA_instruction",
        help="dataset output directory"
    )

    return parser


def process_row(row):
    context_list = row["context.contexts"]
    context = " ".join(context_list).strip()

    row["instruction"] = row["question"]
    row["context"] = context
    row["response"] = row["long_answer"]
    row["category"] = "closed_qa" if len(context) > 0 else "open_qa"

    return row


def process_dataset(
        dataset_name: str,
        config_names: List[str],
        output_path: Union[str, os.PathLike]
):
    dataset_list = []
    for config_name in config_names:
        splits = get_dataset_split_names(dataset_name, config_name)
        assert len(splits) == 1
        split_name = splits[0]
        assert split_name == "train"

        _dataset = load_dataset(dataset_name, config_name, split=split_name)
        dataset_list.append(_dataset)
        del _dataset

    dataset = concatenate_datasets(dataset_list) if len(dataset_list) > 1 else dataset_list[0]
    dataset = dataset.flatten()
    remove_columns = list(set(dataset.column_names).difference(OUTPUT_COLUMNS))
    dataset = dataset.map(
        process_row,
        remove_columns=remove_columns
    )

    output_path = str(output_path)
    output_path = output_path.format(num_rows=dataset.num_rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(output_path))


def main(args: Namespace) -> None:
    # handles relative path
    os.chdir(str(Path(__file__).parent / ".."))

    dataset_name = "pubmed_qa"
    train_config_names = ["pqa_artificial", "pqa_unlabeled"]
    test_config_names = ["pqa_labeled"]
    output_dir = Path(args.output_dir)

    print(f"------------------ train set ------------------")
    process_dataset(
        dataset_name,
        train_config_names,
        output_path=output_dir / f"train_{{num_rows}}.jsonl"
    )

    print(f"------------------ test set ------------------")
    process_dataset(
        dataset_name,
        test_config_names,
        output_path=output_dir / f"test_{{num_rows}}.jsonl"
    )


if __name__ == '__main__':
    main(args=_get_arg_parser().parse_args())
