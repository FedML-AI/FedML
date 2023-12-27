from argparse import ArgumentParser, Namespace
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List

import einops
from huggingface_hub import scan_cache_dir
from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
import torch
from torch import Tensor
from transformers import AutoConfig, LlamaConfig
from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
)
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "../.."))

from src.typing import PathType
from src.utils import get_real_path, is_directory, is_file

KV_NAME_MAPPING = {
    LlamaConfig: ("k_proj", "v_proj"),
}
CONFIG_KEYS_TO_REMOVE = (
    "_name_or_path",
)


def get_hf_checkpoint_shard_files(weight_index_path: PathType) -> List[Path]:
    input_dir = Path(weight_index_path).parent

    with open(get_real_path(weight_index_path), "r") as f:
        weight_index = json.load(f)

    file_paths = [input_dir / filename for filename in set(weight_index["weight_map"].values())]

    if any(not is_file(p) for p in file_paths):
        raise FileNotFoundError(f"{input_dir} contains corrupted checkpoint file.")

    return file_paths


def get_hf_checkpoint_files(input_dir: PathType) -> List[Path]:
    input_dir = Path(input_dir)
    outputs = []

    if is_file(input_dir / WEIGHTS_NAME):
        outputs.append(input_dir / WEIGHTS_NAME)

    if is_file(input_dir / WEIGHTS_INDEX_NAME):
        outputs.extend(get_hf_checkpoint_shard_files(input_dir / WEIGHTS_INDEX_NAME))

    if is_file(input_dir / SAFE_WEIGHTS_NAME):
        outputs.append(input_dir / SAFE_WEIGHTS_NAME)

    if is_file(input_dir / SAFE_WEIGHTS_INDEX_NAME):
        outputs.extend(get_hf_checkpoint_shard_files(input_dir / SAFE_WEIGHTS_INDEX_NAME))

    if len(outputs) == 0:
        raise FileNotFoundError(f"{input_dir} is not a valid hugging face model directory.")

    return outputs


def get_hf_cache_path(model_id: str) -> Path:
    # see https://huggingface.co/docs/huggingface_hub/v0.16.3/en/package_reference/cache
    hf_cache_info = scan_cache_dir()

    for repo in hf_cache_info.repos:
        if repo.repo_id == model_id:
            output = max(repo.revisions, key=lambda x: x.last_modified)
            return output.snapshot_path

    else:
        raise FileNotFoundError(f"Could not find local cache for \"{model_id}\"")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        "--model_name_or_path",
        dest="model_name_or_path",
        help="huggingface model ID or path to model directory.",
        type=str
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="path to output model directory.",
        type=str
    )
    parser.add_argument(
        "-g",
        "--num_groups",
        "--num_kv_heads",
        "--num_attn_groups",
        dest="num_attn_groups",
        help="number of attention KV groups; this is equal to the number of KV heads.",
        type=int
    )
    args = parser.parse_args()
    assert args.num_attn_groups >= 1

    return args


def main(args: Namespace) -> None:
    # add huggingface token if accessing a private model
    # os.environ["HUGGING_FACE_HUB_TOKEN"] = "<huggingface token>"

    if not is_directory(args.model_name_or_path):
        input_dir = get_hf_cache_path(args.model_name_or_path)
    else:
        input_dir = Path(args.model_name_or_path)
    output_dir = Path(args.output_dir)

    assert is_directory(input_dir)
    input_dir = Path(get_real_path(input_dir))
    output_dir = Path(get_real_path(output_dir))

    if is_directory(output_dir) and input_dir != output_dir:
        print(f"Removing existing output directory {output_dir}")
        shutil.rmtree(get_real_path(output_dir))

    config = AutoConfig.from_pretrained(str(input_dir))
    if isinstance(config, LlamaConfig):
        _, remainder = divmod(config.num_key_value_heads, args.num_attn_groups)
        assert remainder == 0, (
            f"The input model has {config.num_key_value_heads:,} key value heads which is not divisible by the"
            f" number of groups {args.num_attn_groups:,}."
        )
        new_config = deepcopy(config)
        new_config.num_key_value_heads = args.num_attn_groups
        if hasattr(new_config, "_name_or_path"):
            delattr(new_config, "_name_or_path")

        prev_num_key_value_heads = config.num_key_value_heads

    else:
        raise ValueError(f"{type(config)} is not a supported HF config type.")

    # get KV param names
    kv_keys = KV_NAME_MAPPING[type(new_config)]

    checkpoint_paths = get_hf_checkpoint_files(input_dir)

    # copy other files
    print("Copying extra files")
    shutil.copytree(
        str(input_dir),
        str(output_dir),
        ignore=shutil.ignore_patterns(
            "*.bin",
            "*.safetensors",
            CONFIG_NAME
        )
    )

    print("Saving config files")
    output_dir.mkdir(parents=True, exist_ok=True)
    new_config.save_pretrained(str(output_dir))

    with open(output_dir / CONFIG_NAME, "r+") as f:
        config_dict = json.load(f)
        # remove keys from the config
        [config_dict.pop(k, None) for k in CONFIG_KEYS_TO_REMOVE]

        # clear file content
        f.seek(0)
        f.truncate()

        # save modified config
        json.dump(config_dict, f, indent=2, sort_keys=True)

    for idx, checkpoint_path in enumerate(checkpoint_paths):
        if checkpoint_path.suffix == ".safetensors":
            checkpoint: Dict[str, Tensor] = safe_load_file(get_real_path(checkpoint_path), "cpu")
        else:
            checkpoint: Dict[str, Tensor] = torch.load(get_real_path(checkpoint_path), "cpu")

        for name, param in tqdm(checkpoint.items(), desc=f"[{idx + 1}/{len(checkpoint_paths)}] Converting checkpoints"):
            if set(name.split(".")).isdisjoint(kv_keys):
                continue

            attn_head_dim = param.shape[0] // prev_num_key_value_heads
            new_attn_dim = attn_head_dim * args.num_attn_groups

            param_groups = einops.rearrange(param, "(h dh) d -> h dh d", dh=new_attn_dim)
            outputs = einops.reduce(param_groups, "h dh d -> dh d", "mean")
            checkpoint[name] = outputs

        output_path = output_dir / checkpoint_path.relative_to(input_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint {output_path}")

        if checkpoint_path.suffix == ".safetensors":
            safe_save_file(checkpoint, str(output_path), metadata={"format": "pt"})
        else:
            torch.save(checkpoint, str(output_path))

        # remove checkpoint from memory
        del checkpoint
        gc.collect()


if __name__ == '__main__':
    main(args=parse_args())
