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
"""

import argparse
import re
import time
from fractions import Fraction
from typing import Optional, List

from datasets import load_dataset          # pip install datasets
from vllm import LLM, SamplingParams       # pip install vllm

# --------------------------- reward configuration ---------------------------

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")  # capture content inside \boxed{…}

EXACT_MATCH_REWARD = 2.0
NUM_EQ_REWARD      = 1.5
INCORRECT_REWARD   = 0.0


# ------------------------------- utilities ---------------------------------

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


# ------------------------------- main --------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=int, default=4,
                   help="completions per example (default: 4)")
    p.add_argument("--batch-examples", type=int, default=16,
                   help="examples per vLLM inference call (default: 2)")
    p.add_argument("--num-examples", type=int, default=100,
                   help="total GSM8K test examples to evaluate (default: 100)")
    p.add_argument(
        "--model",
        default=None,
        help=("HF repo ID or local checkpoint dir. "
              "If omitted, downloads Qwen/Qwen3-0.6B automatically."),
    )

    # When loading a locally fine-tuned checkpoint, the tokenizer files are
    # often *not* included in the output directory.  Allow the user to point
    # to an existing tokenizer (typically the original base model on the HF
    # Hub) to avoid the `vocab_file is None` error coming from
    # `transformers`.
    p.add_argument("--tokenizer",
                   default="Qwen/Qwen3-0.6B",
                   help=("Tokenizer repo / path (default: Qwen/Qwen3-0.6B). "
                         "Override if you need a different tokenizer."))
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="generation length cap (tokens)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------- resolve model path --------------------------
    if args.model is None:
        args.model = "Qwen/Qwen3-0.6B"
        print(f"[INFO] No --model given → downloading '{args.model}' "
              "from Hugging Face Hub…")

    # ----------------------- initialize LLM & sampler ----------------------
    # Use a fallback tokenizer path if the user provided one; otherwise rely on
    # the model path itself.  This prevents crashes when the checkpoint
    # directory does not contain tokenizer artifacts.
    llm = LLM(model=args.model,
              tokenizer=args.tokenizer,
              trust_remote_code=True,   # Qwen uses custom code
              dtype="auto")             # let vLLM choose BF16 / FP16 / FP32

    sampler = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.rollouts
    )

    # --------------------------- load dataset -----------------------------
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42).select(range(min(args.num_examples, len(ds))))

    total_reward = 0.0
    total_completions = len(ds) * args.rollouts

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
            for out in gen.outputs:
                pred = extract_boxed(out.text)
                total_reward += reward(pred, gold)

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


if __name__ == "__main__":
    main()