from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
from typing import Optional, Union
import warnings

# disable GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
)


def get_real_path(path: Union[str, os.PathLike]) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def is_file(path: Union[str, os.PathLike]) -> bool:
    return os.path.isfile(get_real_path(path))


def is_directory(path: Union[str, os.PathLike]) -> bool:
    return os.path.isdir(get_real_path(path))


def verify_hf_model_directory_type(path: Union[str, os.PathLike]) -> Optional[str]:
    path = Path(get_real_path(path))

    if not path.is_dir():
        return None

    if any(is_file(path / filename) for filename in (WEIGHTS_NAME, SAFE_WEIGHTS_NAME)):
        return "base"

    for weights_index_name in (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME):
        weight_index_path = path / weights_index_name
        if is_file(weight_index_path):
            with open(str(weight_index_path), "r") as f:
                weight_index = json.load(f)

            if any(is_file(path / filename) for filename in weight_index["weight_map"].values()):
                # file corrupted
                return None
            else:
                return "base"

    if any(is_file(path / filename) for filename in (ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME)):
        return "peft"

    return None


if __name__ == '__main__':
    # add huggingface token if accessing a private model
    # os.environ["HUGGING_FACE_HUB_TOKEN"] = "<huggingface token>"

    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        "--model_dir",
        dest="input_dir",
        help="path to model directory.",
        type=str
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="path to output model directory.",
        type=str
    )

    args = parser.parse_args()

    model_kwargs = dict(
        trust_remote_code=True,
        # low_cpu_mem_usage=True,
    )
    peft_kwargs = dict(
        is_trainable=True,
    )

    input_dir = Path(get_real_path(args.input_dir))
    output_dir = Path(get_real_path(args.output_dir))

    input_dir_type = verify_hf_model_directory_type(input_dir)
    if input_dir_type is None:
        raise RuntimeError(f"{input_dir} is not a valid huggingface model directory.")

    if input_dir_type == "base":
        warnings.warn("Base model detected. Model checkpoint will be directly copied to the destination directory.")
        if input_dir != output_dir:
            if is_directory(output_dir):
                # remove the destination directory if it's empty
                output_dir.rmdir()
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(input_dir), str(output_dir))
        else:
            warnings.warn("Source and destination are the same. Skipping.")

    elif input_dir_type == "peft":
        print(f"PEFT checkpoint detected. Converting.")
        tokenizer = AutoTokenizer.from_pretrained(str(input_dir), **model_kwargs)
        model = AutoPeftModelForCausalLM.from_pretrained(str(input_dir), **model_kwargs, **peft_kwargs)

        model = model.merge_and_unload()

        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(output_dir))
        model.save_pretrained(str(output_dir))
