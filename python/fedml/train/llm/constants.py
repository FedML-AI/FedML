"""
Adapted from https://github.com/databrickslabs/dolly/blob/master/training/consts.py
"""

# -----------------------------------------------------------------
DEFAULT_MAX_SEQ_LENGTH = 1024
IGNORE_INDEX = -100

# -----------------------------------------------------------------
MODEL_NAMES = [
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "mistralai/Mistral-7B-v0.1"
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
]

DATASET_NAMES = [
    "fedml/PubMedQA_instruction",
    "fedml/databricks-dolly-15k-niid",
    "databricks/databricks-dolly-15k",
    "medalpaca/medical_meadow_mediqa",
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    "gbharti/wealth-alpaca_lora",
    "lavita/ChatDoctor-HealthCareMagic-100k",
]

PEFT_TYPES = [
    "none",
    "lora",
]

PROMPT_STYLES = [
    "default",
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

CUSTOM_LOGGERS = [
    "none",
    "all",
    "fedml",
]
