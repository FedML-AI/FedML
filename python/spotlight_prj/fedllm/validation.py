#!/usr/bin/env python
"""
Evaluate Qwen3‑0.6B on GSM8K (test split) with vLLM.

Usage examples
--------------
# Default: download model, 1 rollout, 8‑question batches, 100 examples
python eval_qwen3_gsm8k.py

# Local checkpoint, 4 rollouts, 16 questions per batch, 200 examples
python eval_qwen3_gsm8k.py \
  --model /path/to/Qwen3-0.6B-local \
  --rollouts 4 \
  --batch-examples 16 \
  --num-examples 200

# Using custom weights with base model config/tokenizer
python eval_qwen3_gsm8k.py \
  --model /path/to/custom_weights.safetensors \
  --base-model Qwen/Qwen3-0.6B \
  --rollouts 4 \
  --batch-examples 16 \
  --num-examples 200
"""

import argparse
import os
import re
import shutil
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Optional, List

from datasets import load_dataset          # pip install datasets
from transformers import AutoConfig, AutoTokenizer  # pip install transformers
from vllm import LLM, SamplingParams       # pip install vllm

# --------------------------- reward configuration ---------------------------

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")  # capture content inside \boxed{…}

EXACT_MATCH_REWARD = 2.0
NUM_EQ_REWARD      = 1.5
INCORRECT_REWARD   = 0.0


# ------------------------------- utilities ---------------------------------

def is_weight_file(path: str) -> bool:
    """Check if path points to a weight file (.bin, .safetensors, .pt, .pth)."""
    if not os.path.isfile(path):
        return False
    return Path(path).suffix.lower() in {'.bin', '.safetensors', '.pt', '.pth'}


def is_complete_checkpoint(path: str) -> bool:
    """Check if path is a directory containing config.json (indicating a complete checkpoint)."""
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, 'config.json'))


def setup_model_with_custom_weights(weight_path: str, base_model: str) -> str:
    """
    Create a temporary directory with base model config/tokenizer and custom weights.
    Returns the path to the temporary directory.
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="qwen_custom_weights_")
    
    try:
        print(f"[INFO] Setting up temporary model directory at {temp_dir}")
        print(f"[INFO] Loading config and tokenizer from base model: {base_model}")
        
        # Download and save config
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        config.save_pretrained(temp_dir)
        
        # Download and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        
        # Copy weight file to temporary directory
        weight_filename = Path(weight_path).name
        dest_weight_path = os.path.join(temp_dir, weight_filename)
        print(f"[INFO] Copying weights from {weight_path} to {dest_weight_path}")
        shutil.copy2(weight_path, dest_weight_path)
        
        print(f"[INFO] Custom model setup complete in {temp_dir}")
        return temp_dir
        
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to setup custom model: {e}")


def to_number(text: str) -> Optional[float]:
    """Convert string to float if possible, handling simple fractions."""
    text = text.replace(",", "").strip()
    # Fractions like 3/4
    if "/" in text:
        try:
            return float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(text)
    except ValueError:
        return None


def extract_boxed(text: str) -> str:
    """Return first \\boxed{...} contents; '' if none."""
    m = BOXED_RE.search(text)
    return m.group(1) if m else ""


def reward(pred: str, gold: str) -> float:
    """Assign reward based on exact match or numeric equivalence."""
    pred, gold = pred.strip(), gold.strip()
    if pred == gold:
        return EXACT_MATCH_REWARD
    p_num, g_num = to_number(pred), to_number(gold)
    if (p_num is not None and g_num is not None
            and abs(p_num - g_num) < 1e-4):
        return NUM_EQ_REWARD
    return INCORRECT_REWARD


def batched(lst: List, n: int):
    """Yield successive n‑sized chunks from *lst*."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_output_filename(model_path: str) -> str:
    """Generate a filesystem-safe filename based on the model path."""
    if model_path is None:
        return "Qwen_Qwen3-0.6B_rewards.csv"
    
    # Extract meaningful name from different model path formats
    if "/" in model_path:
        # HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B") or file path
        name = model_path.split("/")[-1]
        if "." in name:  # Remove file extension for weight files
            name = Path(name).stem
    else:
        name = model_path
    
    # Replace invalid filename characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    
    return f"{name}_rewards.csv"


# ------------------------------- main --------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=int, default=4,
                   help="completions per example (default: 4)")
    p.add_argument("--batch-examples", type=int, default=16,
                   help="examples per vLLM inference call (default: 2)")
    p.add_argument("--num-examples", type=int, default=-1,
                   help="total GSM8K test examples to evaluate (default: 100, use -1 for full dataset)")
    p.add_argument(
        "--model",
        default=None,
        help=("HF repo ID, local checkpoint dir, or path to weight file(s). "
              "If omitted, downloads Qwen/Qwen3-0.6B automatically."),
    )
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen3-0.6B",
        help=("Base model for config/tokenizer when using custom weight files. "
              "Ignored when --model is a full checkpoint directory."),
    )
    p.add_argument("--max-tokens", type=int, default=512,
                   help="generation length cap (tokens)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    temp_model_dir = None

    try:
        # ------------------------- resolve model path --------------------------
        original_model_arg = args.model  # Store original for filename
        if args.model is None:
            args.model = "Qwen/Qwen3-0.6B"
            print(f"[INFO] No --model given → downloading '{args.model}' "
                  "from Hugging Face Hub…")
        elif is_weight_file(args.model):
            print(f"[INFO] Detected weight file: {args.model}")
            print(f"[INFO] Using base model: {args.base_model}")
            temp_model_dir = setup_model_with_custom_weights(args.model, args.base_model)
            args.model = temp_model_dir
        elif is_complete_checkpoint(args.model):
            print(f"[INFO] Using complete checkpoint directory: {args.model}")
        else:
            # Assume it's a HuggingFace model ID
            print(f"[INFO] Using HuggingFace model: {args.model}")

        # ----------------------- initialize LLM & sampler ----------------------
        llm = LLM(model=args.model,
                  trust_remote_code=True,   # Qwen uses custom code
                  dtype="bfloat16")             # let vLLM choose BF16 / FP16 / FP32

        sampler = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.rollouts,
            seed=42
        )

        # --------------------------- load dataset -----------------------------
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if args.num_examples == -1:
            # Use full dataset without shuffling
            print(f"[INFO] Using full dataset ({len(ds)} examples)")
        else:
            # Shuffle and select specified number of examples
            num_to_select = min(args.num_examples, len(ds))
            ds = ds.shuffle(seed=42).select(range(num_to_select))
            print(f"[INFO] Using {len(ds)} examples (shuffled)")

        total_reward = 0.0
        total_completions = len(ds) * args.rollouts
        all_example_rewards = []  # Track all rollout rewards for each example

        # --------------------------- evaluation -------------------------------
        print(f"[INFO] Starting generation for {len(ds)} examples in batches of {args.batch_examples}...")
        start_time = time.time()
        
        batch_count = 0
        for batch in batched(list(ds), args.batch_examples):
            batch_start = time.time()
            prompts = [ex["question"] for ex in batch]   # **raw questions only**
            outputs = llm.generate(prompts, sampler)
            batch_end = time.time()
            
            batch_count += 1
            batch_time = batch_end - batch_start
            print(f"[TIMING] Batch {batch_count} ({len(batch)} examples): {batch_time:.2f}s")

            for ex, gen in zip(batch, outputs):
                gold = ex["answer"].split("####")[-1].strip()
                example_rollout_rewards = []
                for out in gen.outputs:
                    pred = extract_boxed(out.text)
                    rollout_reward = reward(pred, gold)
                    total_reward += rollout_reward
                    example_rollout_rewards.append(rollout_reward)
                
                # Store all rollout rewards for this example
                all_example_rewards.append(example_rollout_rewards)

        end_time = time.time()
        total_generation_time = end_time - start_time
        
        avg_reward = total_reward / total_completions
        print(f"\n[TIMING] Total generation time: {total_generation_time:.2f}s")
        print(f"[TIMING] Average time per batch: {total_generation_time / batch_count:.2f}s")
        print(f"[TIMING] Average time per example: {total_generation_time / len(ds):.2f}s")
        print(f"[TIMING] Average time per completion: {total_generation_time / total_completions:.3f}s")
        print(f"\nEvaluated {len(ds)} examples × {args.rollouts} rollouts "
              f"(batch size = {args.batch_examples}).")
        print(f"Average reward: {avg_reward:.4f}")

        # ------------------------ write rewards to file -------------------------
        output_filename = get_output_filename(original_model_arg)
        
        with open(output_filename, 'w') as f:
            for example_rewards in all_example_rewards:
                line = ",".join(str(r) for r in example_rewards)
                f.write(line + "\n")
        
        print(f"[INFO] Individual example rewards written to: {output_filename}")
        print(f"[INFO] File contains {len(all_example_rewards)} lines, one per example")
        
        # Calculate max reward per example for summary stats
        max_rewards_per_example = [max(rewards) for rewards in all_example_rewards]
        print(f"[INFO] Max reward per example average: {sum(max_rewards_per_example) / len(max_rewards_per_example):.4f}")

    finally:
        # Clean up temporary directory if it was created
        if temp_model_dir and os.path.exists(temp_model_dir):
            print(f"[INFO] Cleaning up temporary directory: {temp_model_dir}")
            shutil.rmtree(temp_model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()