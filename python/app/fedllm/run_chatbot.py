from timeit import default_timer as timer

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from src.instruct_pipeline import InstructionTextGenerationPipeline

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        "--model_name_or_path",
        dest="model_name_or_path",
        type=str,
        required=True,
        help="model name or path to model checkpoint directory"
    )
    args = parser.parse_args()

    print("Initializing...")
    st_time = init_st_time = timer()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, device_map="auto")

    instruct_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        config=config,
        pipeline_class=InstructionTextGenerationPipeline
    )
    print(type(instruct_pipeline))
    print(f"Initialization took {timer() - init_st_time:,.2f}s")

    print("Running...")
    while True:
        question = input("Enter your question or enter \"exit\" to quit: ")

        if question.lower().strip() == "exit":
            break

        print(f"Trying to answer your question:\n\"{question}\"\n")
        gen_st_time = timer()
        response = instruct_pipeline(question)
        print(f"Response:\n\"{response[0]['generated_text']}\"\n")
        gen_end_time = timer()

        print(f"Inference took {gen_end_time - gen_st_time:,.2f}s\n")
