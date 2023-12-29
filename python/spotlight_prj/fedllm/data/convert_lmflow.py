from typing import Any, Dict, List

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path

from tqdm import tqdm

STR_REPLACE = {
    "Definition:": "",
    "Input:": "",
    "Output:": "",
    "Question:": ""
}


def _get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--dataset_path",
        type=str,
        required=True,
        metavar="DF",
        help="dataset path",
        nargs="+"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        metavar="O",
        help="dataset output path",
        nargs="+"
    )

    return parser


def replace_multiple_str(input_str: str) -> str:
    for k, v in STR_REPLACE.items():
        # Replace key character with value character in string
        input_str = input_str.replace(k, v)

    return input_str


def convert_data(data_dict: Dict[str, str]) -> Dict[str, str]:
    if {"input", "output"}.issubset(data_dict):
        output_dict = {
            "instruction": replace_multiple_str(data_dict["input"]).strip(),
            "context": "",
            "response": replace_multiple_str(data_dict["output"]).strip(),
            "category": "classification"
        }

    else:
        output_dict = {
            "instruction": "",
            "context": "",
            "response": "",
            "category": "classification"
        }

        groups = []
        for idx, s in enumerate(data_dict["text"].splitlines()):
            s = s.strip().replace("Defintion:", "Definition:")

            if len(s) == 0:
                continue

            elif not s.startswith(("Definition:", "Input:", "Output:")):
                assert idx > 0
                assert len(groups) > 0
                groups[-1] = f"{groups[-1]}\n{s}"

            else:
                groups.append(s)

        for s in groups:
            if len(s) == 0 or s.startswith("Definition:"):
                continue

            parsed_s = replace_multiple_str(s).strip()
            if s.startswith("Input:"):
                output_dict["instruction"] = parsed_s
            elif s.startswith("Output:"):
                output_dict["response"] = parsed_s
            else:
                raise ValueError(
                    f"string must start with on of the following prefix {list(STR_REPLACE.keys())},"
                    f" but got prefix \"{s[:max(len(k) for k in STR_REPLACE)]}\""
                )

    return output_dict


def main(args: Namespace) -> None:
    assert len(args.dataset_path) == len(args.output_path)

    for idx, (input_path, output_path) in enumerate(zip(args.dataset_path, args.output_path)):
        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "r") as f, open(output_path, "w", encoding="utf-8") as of:
            data_dict: Dict[str, Any] = json.load(f)
            data_samples: List[Dict[str]] = data_dict["instances"]

            for data_sample in tqdm(data_samples, desc=f"[{idx + 1}/{len(args.dataset_path)}] Processing"):
                print(f"{json.dumps(convert_data(data_sample))}", file=of)
                del data_sample
            del data_dict, data_samples


if __name__ == '__main__':
    main(args=_get_arg_parser().parse_args())
