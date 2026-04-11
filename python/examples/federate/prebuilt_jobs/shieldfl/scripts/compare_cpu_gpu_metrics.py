"""
对比 CPU 和 GPU 运行的结构化指标文件，输出对齐报告。

对齐判定规则（对应 PHASE2_GPU.md §5 Step 4）：
  - FedAvg:  前 N 轮 |Δ test_accuracy| ≤ tolerance（默认 0.005 即 0.5%）
  - VeriFL:  前 N 轮 |Δ test_accuracy| ≤ tolerance（默认 0.01 即 1.0%）

用法：
  python scripts/compare_cpu_gpu_metrics.py \\
      --cpu results/alignment_cpu/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed0.jsonl \\
      --gpu results/alignment_gpu/metrics_ResNet18_cifar10_fedavg_atknone_defnone_seed0.jsonl \\
      --tolerance 0.005 \\
      --label FedAvg

退出码：
  0 — 对齐通过（所有轮次差异 ≤ tolerance）
  1 — 对齐失败（存在超出 tolerance 的轮次）
  2 — 文件加载或格式错误
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """从 .jsonl 文件中加载逐轮指标，按 round 字段升序排列。"""
    records: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error at {path}:{lineno}: {exc}") from exc
    records.sort(key=lambda r: r.get("round", 0))
    return records


def compare(
    cpu_records: List[Dict[str, Any]],
    gpu_records: List[Dict[str, Any]],
    tolerance: float,
    label: str,
) -> bool:
    """
    逐轮对比 test_accuracy 和 test_loss，打印报告，返回是否全部通过。
    """
    cpu_by_round = {r["round"]: r for r in cpu_records}
    gpu_by_round = {r["round"]: r for r in gpu_records}

    common_rounds = sorted(set(cpu_by_round) & set(gpu_by_round))
    if not common_rounds:
        print(f"[{label}] WARNING: no common rounds found between CPU and GPU files.")
        return False

    print(f"\n{'='*60}")
    print(f"[{label}] CPU vs GPU Alignment Report  (tolerance={tolerance:.4f})")
    print(f"{'='*60}")
    print(f"{'Round':>6}  {'CPU_MA':>8}  {'GPU_MA':>8}  {'|ΔMA|':>8}  {'Status':>8}")
    print(f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    passed = True
    for rnd in common_rounds:
        cpu_acc = float(cpu_by_round[rnd].get("test_accuracy", float("nan")))
        gpu_acc = float(gpu_by_round[rnd].get("test_accuracy", float("nan")))
        delta = abs(cpu_acc - gpu_acc)
        ok = delta <= tolerance
        status = "PASS" if ok else "FAIL"
        if not ok:
            passed = False
        print(f"{rnd:>6}  {cpu_acc:>8.4f}  {gpu_acc:>8.4f}  {delta:>8.4f}  {status:>8}")

    # loss 对比（仅展示，不参与 pass/fail 判定）
    print(f"\n{'Round':>6}  {'CPU_Loss':>10}  {'GPU_Loss':>10}  {'|ΔLoss|':>10}")
    print(f"{'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    for rnd in common_rounds:
        cpu_loss = float(cpu_by_round[rnd].get("test_loss", float("nan")))
        gpu_loss = float(gpu_by_round[rnd].get("test_loss", float("nan")))
        delta_loss = abs(cpu_loss - gpu_loss)
        print(f"{rnd:>6}  {cpu_loss:>10.6f}  {gpu_loss:>10.6f}  {delta_loss:>10.6f}")

    # 硬件上下文展示
    cpu_device = cpu_records[0].get("device", "unknown") if cpu_records else "unknown"
    gpu_device = gpu_records[0].get("device", "unknown") if gpu_records else "unknown"
    cpu_mode = cpu_records[0].get("runtime_mode", "unknown") if cpu_records else "unknown"
    gpu_mode = gpu_records[0].get("runtime_mode", "unknown") if gpu_records else "unknown"
    print(f"\nCPU run: device={cpu_device!r}  runtime_mode={cpu_mode!r}")
    print(f"GPU run: device={gpu_device!r}  runtime_mode={gpu_mode!r}")

    verdict = "PASSED" if passed else "FAILED"
    print(f"\n[{label}] Alignment {verdict} (tolerance={tolerance:.4f}, rounds={common_rounds})\n")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CPU vs GPU ShieldFL metrics files for numerical alignment."
    )
    parser.add_argument("--cpu", required=True, help="Path to CPU .jsonl metrics file")
    parser.add_argument("--gpu", required=True, help="Path to GPU .jsonl metrics file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Maximum allowed |Δ test_accuracy| per round (default: 0.005 = 0.5%%)",
    )
    parser.add_argument("--label", default="Alignment", help="Label for the report header")
    args = parser.parse_args()

    try:
        cpu_records = load_metrics(args.cpu)
        gpu_records = load_metrics(args.gpu)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ok = compare(cpu_records, gpu_records, args.tolerance, args.label)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
