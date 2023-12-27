import os
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from typing import Optional

from huggingface_hub import HfApi
from transformers.utils import CONFIG_NAME

sys.path.append(str(Path(__file__).parent / "../.."))

from src.utils import is_directory, is_file

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        dest="input_dir",
        help="path to model directory.",
        type=str
    )
    parser.add_argument(
        "-o",
        "--repo_id",
        "--model_id",
        "--model_repo_id",
        dest="repo_id",
        help="huggingface model repo ID.",
        type=str
    )
    parser.add_argument(
        "-b",
        "--branch",
        dest="branch",
        help="huggingface model repo branch.",
        type=str,
        default=None
    )
    parser.add_argument(
        "-k",
        "--token",
        dest="token",
        help="huggingface token.",
        type=str,
        default=None
    )
    parser.add_argument(
        "-m",
        "--commit_message",
        dest="commit_message",
        help="commit message.",
        type=str,
        default=None
    )
    args = parser.parse_args()

    # unpack args
    input_dir: str = args.input_dir
    repo_id: str = args.repo_id
    branch: Optional[str] = args.branch
    token: Optional[str] = args.token
    commit_message: Optional[str] = args.commit_message

    # verify args
    assert is_directory(input_dir)
    if not bool(branch):
        branch = None
    if not bool(token):
        token = os.getenv("HUGGING_FACE_HUB_TOKEN", token)
        token = None if not bool(token) else token
    if not bool(commit_message):
        commit_message = None

    input_dir: Path = Path(input_dir)
    config_path = input_dir / CONFIG_NAME
    assert is_file(config_path)

    print("Updating config")
    with open(config_path, "r+") as f:
        config_dict = json.load(f)
        config_dict["_name_or_path"] = repo_id

        # clear file content
        f.seek(0)
        f.truncate()

        # save modified config
        json.dump(config_dict, f, indent=2, sort_keys=True)

    api = HfApi(token=token)

    print("Setting up repo")
    api.create_repo(
        repo_id=repo_id,
        private=True
    )
    if bool(branch):
        api.create_branch(
            repo_id=repo_id,
            branch=branch,
            exist_ok=True
        )
    print("Uploading")
    api.upload_folder(
        folder_path=input_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        revision=branch
    )
