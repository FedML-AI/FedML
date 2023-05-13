"""
See https://www.deepspeed.ai/tutorials/inference-tutorial/ for detail
"""
import os
import sys

import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import to_device

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        "--model_name_or_path",
        dest="model_name_or_path",
        type=str,
        default="EleutherAI/pythia-70m",
        help="model name or path to model checkpoint directory"
    )
    parser.add_argument("--deepspeed", dest="deepspeed", action='store_true', default=True)
    parser.add_argument("--no-deepspeed", "--no_deepspeed", dest='deepspeed', action='store_false')
    args, _ = parser.parse_known_args()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = 0

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.deepspeed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # Initialize the DeepSpeed-Inference engine
        ds_engine = deepspeed.init_inference(
            model,
            mp_size=world_size,
            dtype=torch.float,
            replace_with_kernel_inject=True
        )
        model = ds_engine.module

        # update rank
        rank = deepspeed.comm.get_rank()
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        model.to(device)
    else:
        device = torch.device("cpu")

    input_str = "hello, I am a robot"
    batch = tokenizer(input_str, return_tensors="pt")

    input_ids = to_device(batch["input_ids"], device)
    attention_mask = to_device(batch.get("attention_mask", None), device)

    generated_sequence = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        pad_token_id=tokenizer.pad_token_id)

    if rank == 0:
        for sequence in generated_sequence.tolist():
            print(tokenizer.decode(sequence).strip())
