#!/usr/bin/env python3
"""Save initial checkpoint for FedML server."""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
RUN_ID = os.environ.get("RUN_ID", "test_run")
MODEL_NAME = "Qwen/Qwen3-1.7B-GPTQ-Int8"
OUTPUT_DIR = f"/workspace/FedML/python/spotlight_prj/fedllm/.logs/FedML/{RUN_ID}/node_0/init"

print(f"Saving initial checkpoint for model: {MODEL_NAME}")
print(f"Output directory: {OUTPUT_DIR}")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load model and tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Save model in the format expected by FedML (pytorch_model.bin)
print("Saving model checkpoint...")
# Save using safe_serialization=False to get pytorch_model.bin instead of model.safetensors
model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Initial checkpoint saved to {OUTPUT_DIR}")
print("Files created:")
for file in Path(OUTPUT_DIR).iterdir():
    print(f"  - {file.name}") 