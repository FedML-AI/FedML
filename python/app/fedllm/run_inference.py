"""
See https://www.deepspeed.ai/tutorials/inference-tutorial/ for detail
"""
import os

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
    parser.add_argument(
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=256,
        help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
    )
    parser.add_argument("--deepspeed", dest="deepspeed", action="store_true", default=True)
    parser.add_argument("--no-deepspeed", "--no_deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--no-sample", "--no_sample", dest='do_sample', action="store_false")
    args, _ = parser.parse_known_args()
    assert args.max_new_tokens > 0

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
            dtype=torch.float32,
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

    input_str = "DeepSpeed is"
    batch = tokenizer(input_str, return_tensors="pt")

    input_ids = to_device(batch["input_ids"], device)
    attention_mask = to_device(batch.get("attention_mask", None), device)

    generated_sequence = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens
    )

    if rank == 0:
        print(f"Prompt: \"{input_str}\"")
        for sequence in generated_sequence.tolist():
            print(f"Response: \"{tokenizer.decode(sequence).strip()}\"")
