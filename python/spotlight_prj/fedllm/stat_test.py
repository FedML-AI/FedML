#!/usr/bin/env python3
"""
perm_test_files.py  – Paired permutation test for model rewards (multi-rollout format)

File format
-----------
Each line = one evaluation question.
Each line contains comma-separated rewards (0, 1.5, 2) – one per rollout.

Example (Three questions, two rollouts each):
2,1.5
2,2
0,0

Usage
-----
python perm_test_files.py <modelA_file> <modelB_file>
"""
from __future__ import annotations
import sys
import pathlib
import numpy as np
from typing import Callable

# ----------------------------------------------------------------------
# CONFIGURATION: choose how to collapse multiple roll-outs into one value.
# ----------------------------------------------------------------------
AGG_FUNC: Callable[[np.ndarray], float] = np.mean      # or np.max, etc.

# ----------------------------------------------------------------------
def load_rewards(path: str | pathlib.Path) -> np.ndarray:
    """
    Read *path* and return a 1-D array of per-question aggregated rewards.
    Each line is split on commas, converted to floats, then collapsed with AGG_FUNC.
    """
    try:
        lines = pathlib.Path(path).read_text().strip().splitlines()
    except OSError as err:
        sys.exit(f"Error reading '{path}': {err}")

    if not lines:
        sys.exit(f"Error: '{path}' is empty.")

    per_question = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            sys.exit(f"Error: blank line at {path}:{lineno}.")
        try:
            values = np.fromstring(line, sep=",", dtype=float)
        except ValueError as err:
            sys.exit(f"Error parsing numbers in '{path}' line {lineno}: {err}")
        if values.size == 0:
            sys.exit(f"Error: no numeric values in '{path}' line {lineno}.")
        per_question.append(AGG_FUNC(values))

    return np.asarray(per_question, dtype=float)


def permutation_test(rA: np.ndarray,
                     rB: np.ndarray,
                     B: int = 100_000,
                     seed: int = 42) -> tuple[float, float]:
    """
    Paired permutation test on per-question reward differences.

    Returns
    -------
    gap : float         mean(rA − rB)
    p_two_sided : float permutation p-value
    """
    d   = rA - rB
    gap = d.mean()

    rng = np.random.default_rng(seed)
    signs = rng.choice([1, -1], size=(B, d.size))
    perm_gaps = (signs * d).mean(axis=1)
    p_two_sided = (np.abs(perm_gaps) >= abs(gap)).mean()
    return gap, p_two_sided


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage:  python perm_test_files.py <modelA_file> <modelB_file>")
        sys.exit(1)

    rA = load_rewards(sys.argv[1])
    rB = load_rewards(sys.argv[2])

    if rA.size != rB.size:
        sys.exit("Error: the two files contain different numbers of questions.")

    gap, p = permutation_test(rA, rB)

    print(f"# questions                   : {rA.size}")
    print(f"Aggregation over roll-outs    : {AGG_FUNC.__name__}")
    print(f"Mean reward difference (A-B)  : {gap:.6f}")
    print(f"Two-sided permutation p-value : {p:.6g}")


if __name__ == "__main__":
    main()