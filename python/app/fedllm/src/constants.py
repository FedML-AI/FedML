"""
Adapted from https://github.com/databrickslabs/dolly/blob/master/training/consts.py
"""

# -----------------------------------------------------------------
DEFAULT_MAX_SEQ_LENGTH = 1024
IGNORE_INDEX = -100

# -----------------------------------------------------------------
FINETUNE_TASKS = [
    "finetune",
    "instruction",
]

# -----------------------------------------------------------------
MODEL_NAMES = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-j-6B",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
]

DATASET_NAMES = [
    "databricks/databricks-dolly-15k",
    "togethercomputer/RedPajama-Data-1T",
    "togethercomputer/RedPajama-Data-1T-Sample",
    "EleutherAI/pile",
    # "EleutherAI/pile" is no longer available for some unknown reasons. The below datasets are backups.
    "FedML/pile",
    "medalpaca/medical_meadow_mediqa",
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    "gbharti/wealth-alpaca_lora",
]

PROMPT_STYLES = [
    "dolly",
    "llama",
]

MODEL_DTYPE_MAPPING = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": None,
    "float": None,
    "float32": None,
    "none": None,
}
MODEL_DTYPES = list(MODEL_DTYPE_MAPPING.keys())
