"""
Adapted from https://github.com/databrickslabs/dolly/blob/master/training/consts.py
"""
import torch

# -----------------------------------------------------------------
DEFAULT_MAX_SEQ_LENGTH = 1024
IGNORE_INDEX = -100

# -----------------------------------------------------------------
MODEL_NAMES = [
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "stabilityai/stablelm-tuned-alpha-7b",
    "OpenAssistant/oasst-sft-1-pythia-12b",
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "OpenAssistant/llama2-13b-orca-8k-3319",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "HuggingFaceH4/mistral-7b-sft-alpha",
    "HuggingFaceH4/mistral-7b-sft-beta",
    "HuggingFaceH4/zephyr-7b-alpha",
    "HuggingFaceH4/zephyr-7b-beta",
]

DEFAULT_MODEL_KWARGS = {
    "torch_dtype": torch.float16,
    "device_map": "auto",
}
DEFAULT_HF_PIPELINE_KWARGS = {
    "temperature": 0.5,
    "max_new_tokens": DEFAULT_MAX_SEQ_LENGTH,
    "do_sample": True,
}

MODEL_DTYPES = ["auto", "fp32", "fp16", "bf16", "4bit", "8bit"]
