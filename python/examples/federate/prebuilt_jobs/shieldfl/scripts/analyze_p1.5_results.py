#!/usr/bin/env python3
"""P1.5 Scaling Attack Verification - Results Analysis
Based on M2_SA_FIX_PHASE1_ERROR_FIX.md AC criteria.
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Configuration
RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results"
CTRL_DIR = os.path.join(RESULTS_DIR, "p1.5_gamma1_ctrl")
AUTO_BACKUP_DIR = os.path.join(RESULTS_DIR, "p1.5_gamma_auto_ctrl_backup")

# P1.5 experiment matrix
CIFAR_ALPHAS = [0.1, 0.3, 0.5, 100]
MNIST_ALPHAS = [0.1, 0.3, 0.5, 100]
CIFAR_SEEDS = [0, 1, 2]
MNIST_SEEDS = [0, 1, 2, 3, 4]
PMR = 0.1

# M1.5 clean baselines (from M1.5_EXPERIMENT_REPORT.md)
M15_BASELINES = {
    ("cifar10", 0.1): {0: 0.6007, 1: 0.4330, 2: 0.5989},
    ("cifar10", 0.3): {0: 0.7580, 1: 0.7212, 2: 0.7399},
    ("cifar10", 0.5): {0: 0.7946, 1: 0.7946, 2: 0.7905},
    ("cifar10", 100):  {0: 0.8209, 1: 0.8210, 2: 0.8214},
    ("mnist", 0.1): {0: 0.9739, 1: 0.9821, 2: 0.9704},
    ("mnist", 0.3): {0: 0.9788, 1: 0.9859, 2: 0.9854},
    ("mnist", 0.5): {0: 0.9827, 1: 0.9893, 2: 0.9879},
    ("mnist", 100):  {0: 0.9906, 1: 0.9908, 2: 0.9898},
}


def load_jsonl(filepath):
    """Load JSONL file and return list of dicts."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_final_round(records, dataset):
    """Get the final round metrics (last occurrence, to handle appended JSONL)."""
    target_round = 99 if dataset == "cifar10" else 49
    result = None
    for r in records:
        if r.get("round") == target_round:
            result = r  # keep last match
    if result:
        return result
    # Fallback: last record
    return records[-1] if records else None


def find_main_results():
    """Find all main experiment (γ=auto) JSONL files."""
    results = {}

    # CIFAR-10
    for alpha in CIFAR_ALPHAS:
        for seed in CIFAR_SEEDS:
            alpha_str = str(int(alpha)) if alpha == int(alpha) and alpha >= 1 else str(alpha)
            fname = f"metrics_ResNet18_cifar10_shieldfl_atkmodel_replacement_defnone_a{alpha_str}_pmr{PMR}_seed{seed}.jsonl"
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                results[("cifar10", alpha, seed)] = fpath
            else:
                # Check backup dir (for α=0.5/100 seed=0 that may have been overwritten by control)
                backup = os.path.join(AUTO_BACKUP_DIR, fname)
                if os.path.exists(backup):
                    results[("cifar10", alpha, seed)] = backup

    # MNIST
    for alpha in MNIST_ALPHAS:
        for seed in MNIST_SEEDS:
            alpha_str = str(int(alpha)) if alpha == int(alpha) and alpha >= 1 else str(alpha)
            fname = f"metrics_LeNet5_mnist_shieldfl_atkmodel_replacement_defnone_a{alpha_str}_pmr{PMR}_seed{seed}.jsonl"
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                results[("mnist", alpha, seed)] = fpath
            else:
                backup = os.path.join(AUTO_BACKUP_DIR, fname)
                if os.path.exists(backup):
                    results[("mnist", alpha, seed)] = backup

    return results


def find_control_results():
    """Find γ=1 control experiment JSONL files."""
    controls = {}
    expected = [
        ("cifar10", 0.5, 0, "metrics_ResNet18_cifar10_ctrl_g1_a0.5_seed0.jsonl"),
        ("cifar10", 100, 0, "metrics_ResNet18_cifar10_ctrl_g1_a100_seed0.jsonl"),
        ("mnist", 0.5, 0, "metrics_LeNet5_mnist_ctrl_g1_a0.5_seed0.jsonl"),
    ]
    for dataset, alpha, seed, fname in expected:
        fpath = os.path.join(CTRL_DIR, fname)
        if os.path.exists(fpath):
            controls[(dataset, alpha, seed)] = fpath
    return controls


def analyze():
    print("=" * 80)
    print("P1.5 Scaling Attack Verification - Results Analysis")
    print("=" * 80)

    main_results = find_main_results()
    ctrl_results = find_control_results()

    print(f"\nFound {len(main_results)} main experiments, {len(ctrl_results)} control experiments")
    print(f"Expected: 32 main (12 CIFAR + 20 MNIST), 3 control")

    # Parse all results
    main_data = {}
    for key, fpath in main_results.items():
        records = load_jsonl(fpath)
        # If file has appended runs, only use the last run (last N rounds)
        target_rounds = 100 if key[0] == "cifar10" else 50
        if len(records) > target_rounds:
            records = records[-target_rounds:]
        final = get_final_round(records, key[0])
        if final:
            main_data[key] = {
                "acc": final.get("test_accuracy", 0),
                "asr": final.get("asr", 0),
                "loss": final.get("test_loss", 0),
                "has_nan": any(
                    r.get("test_loss") is not None and
                    (str(r.get("test_loss")) == "nan" or str(r.get("test_loss")) == "NaN")
                    for r in records
                ),
                "records": records,
            }

    ctrl_data = {}
    for key, fpath in ctrl_results.items():
        records = load_jsonl(fpath)
        final = get_final_round(records, key[0])
        if final:
            ctrl_data[key] = {
                "acc": final.get("test_accuracy", 0),
                "asr": final.get("asr", 0),
                "loss": final.get("test_loss", 0),
            }

    # === CIFAR-10 Results Table ===
    print("\n" + "=" * 80)
    print("CIFAR-10 Results (γ=auto, attack=[99])")
    print("=" * 80)
    print(f"{'α':<8} {'seed':>4} {'Final Acc':>10} {'Final ASR':>10} {'Final Loss':>11} {'NaN?':>5}")
    print("-" * 55)

    cifar_asr_by_alpha = defaultdict(list)
    cifar_acc_by_alpha = defaultdict(list)
    cifar_nan_count = 0

    for alpha in CIFAR_ALPHAS:
        for seed in CIFAR_SEEDS:
            key = ("cifar10", alpha, seed)
            if key in main_data:
                d = main_data[key]
                nan_flag = "YES" if d["has_nan"] else "no"
                if d["has_nan"]:
                    cifar_nan_count += 1
                print(f"{alpha:<8} {seed:>4} {d['acc']:>10.4f} {d['asr']:>10.4f} {d['loss']:>11.4f} {nan_flag:>5}")
                cifar_asr_by_alpha[alpha].append(d["asr"])
                cifar_acc_by_alpha[alpha].append(d["acc"])
            else:
                print(f"{alpha:<8} {seed:>4} {'MISSING':>10}")

    # CIFAR-10 per-alpha summary
    print("\nCIFAR-10 Per-α Summary:")
    print(f"{'α':<8} {'Mean ASR':>10} {'Seeds≥0.80':>11} {'Mean Acc':>10} {'Acc>0.30':>9}")
    print("-" * 55)
    cifar_alpha_pass_asr = 0
    for alpha in CIFAR_ALPHAS:
        asrs = cifar_asr_by_alpha[alpha]
        accs = cifar_acc_by_alpha[alpha]
        if asrs:
            mean_asr = sum(asrs) / len(asrs)
            seeds_above = sum(1 for a in asrs if a >= 0.80)
            mean_acc = sum(accs) / len(accs)
            acc_above = sum(1 for a in accs if a > 0.30)
            pass_mark = "✅" if mean_asr >= 0.80 else "❌"
            print(f"{alpha:<8} {mean_asr:>10.4f} {seeds_above:>9}/3  {mean_acc:>10.4f} {acc_above:>7}/3  {pass_mark}")
            if mean_asr >= 0.80:
                cifar_alpha_pass_asr += 1

    # === MNIST Results Table ===
    print("\n" + "=" * 80)
    print("MNIST Results (γ=auto, attack=[49])")
    print("=" * 80)
    print(f"{'α':<8} {'seed':>4} {'Final Acc':>10} {'Final ASR':>10} {'Final Loss':>11} {'NaN?':>5}")
    print("-" * 55)

    mnist_asr_by_alpha = defaultdict(list)
    mnist_acc_by_alpha = defaultdict(list)
    mnist_nan_count = 0

    for alpha in MNIST_ALPHAS:
        for seed in MNIST_SEEDS:
            key = ("mnist", alpha, seed)
            if key in main_data:
                d = main_data[key]
                nan_flag = "YES" if d["has_nan"] else "no"
                if d["has_nan"]:
                    mnist_nan_count += 1
                print(f"{alpha:<8} {seed:>4} {d['acc']:>10.4f} {d['asr']:>10.4f} {d['loss']:>11.4f} {nan_flag:>5}")
                mnist_asr_by_alpha[alpha].append(d["asr"])
                mnist_acc_by_alpha[alpha].append(d["acc"])
            else:
                print(f"{alpha:<8} {seed:>4} {'MISSING':>10}")

    # MNIST per-alpha summary (median-based)
    print("\nMNIST Per-α Summary (5 seeds, median-based):")
    print(f"{'α':<8} {'Median ASR':>11} {'Mean ASR':>10} {'Seeds≥0.80':>11}")
    print("-" * 50)
    mnist_alpha_pass = 0
    for alpha in MNIST_ALPHAS:
        asrs = sorted(mnist_asr_by_alpha[alpha])
        if asrs:
            median_asr = asrs[len(asrs) // 2]
            mean_asr = sum(asrs) / len(asrs)
            seeds_above = sum(1 for a in asrs if a >= 0.80)
            pass_mark = "✅" if median_asr >= 0.80 else "❌"
            print(f"{alpha:<8} {median_asr:>11.4f} {mean_asr:>10.4f} {seeds_above:>9}/{len(asrs)}  {pass_mark}")
            if median_asr >= 0.80:
                mnist_alpha_pass += 1

    # === Control Group ===
    print("\n" + "=" * 80)
    print("Control Group (γ=1)")
    print("=" * 80)
    print(f"{'Dataset':<10} {'α':<8} {'seed':>4} {'γ=1 ASR':>10} {'γ=auto ASR':>11} {'ΔASR':>8}")
    print("-" * 55)

    delta_asrs = []
    for dataset, alpha, seed in [("cifar10", 0.5, 0), ("cifar10", 100, 0), ("mnist", 0.5, 0)]:
        ctrl_key = (dataset, alpha, seed)
        main_key = (dataset, alpha, seed)
        g1_asr = ctrl_data.get(ctrl_key, {}).get("asr", "N/A")
        auto_asr = main_data.get(main_key, {}).get("asr", "N/A")

        if isinstance(g1_asr, (int, float)) and isinstance(auto_asr, (int, float)):
            delta = auto_asr - g1_asr
            delta_asrs.append(delta)
            print(f"{dataset:<10} {alpha:<8} {seed:>4} {g1_asr:>10.4f} {auto_asr:>11.4f} {delta:>8.4f}")
        else:
            print(f"{dataset:<10} {alpha:<8} {seed:>4} {str(g1_asr):>10} {str(auto_asr):>11}")

    # === Clean Accuracy Drop ===
    print("\n" + "=" * 80)
    print("Clean Accuracy Drop vs M1.5 Baseline")
    print("=" * 80)

    print("\nCIFAR-10 Clean Drop (pp):")
    print(f"{'α':<8} {'seed':>4} {'Baseline':>10} {'Attack':>10} {'Drop(pp)':>10}")
    print("-" * 50)
    cifar_drop_by_alpha = defaultdict(list)
    for alpha in CIFAR_ALPHAS:
        for seed in CIFAR_SEEDS:
            bl_key = ("cifar10", alpha)
            main_key = ("cifar10", alpha, seed)
            if bl_key in M15_BASELINES and seed in M15_BASELINES[bl_key] and main_key in main_data:
                bl = M15_BASELINES[bl_key][seed]
                atk = main_data[main_key]["acc"]
                drop = (bl - atk) * 100  # pp
                cifar_drop_by_alpha[alpha].append(drop)
                print(f"{alpha:<8} {seed:>4} {bl:>10.4f} {atk:>10.4f} {drop:>10.2f}")

    print("\nMNIST Clean Drop (pp):")
    print(f"{'α':<8} {'seed':>4} {'Baseline':>10} {'Attack':>10} {'Drop(pp)':>10}")
    print("-" * 50)
    mnist_drop_by_alpha = defaultdict(list)
    for alpha in MNIST_ALPHAS:
        for seed in [0, 1, 2]:  # baselines only for seeds 0-2
            bl_key = ("mnist", alpha)
            main_key = ("mnist", alpha, seed)
            if bl_key in M15_BASELINES and seed in M15_BASELINES[bl_key] and main_key in main_data:
                bl = M15_BASELINES[bl_key][seed]
                atk = main_data[main_key]["acc"]
                drop = (bl - atk) * 100
                mnist_drop_by_alpha[alpha].append(drop)
                print(f"{alpha:<8} {seed:>4} {bl:>10.4f} {atk:>10.4f} {drop:>10.2f}")

    # === AC Verdict ===
    print("\n" + "=" * 80)
    print("AC VERDICT")
    print("=" * 80)

    verdicts = {}

    # AC-C-1: auto-gamma日志格式 (需人工检查日志)
    verdicts["AC-C-1"] = ("CHECK", "需要检查实验日志中 'Scaling auto-gamma' 行")

    # AC-C-2: 固定γ行为无regression (需对比P1 γ=1结果)
    verdicts["AC-C-2"] = ("CHECK", "需对比P1 γ=1控制组结果")

    # AC-C-3: γ=1不触发auto路径
    verdicts["AC-C-3"] = ("CHECK", "需检查控制组日志无 'auto-gamma' 输出")

    # AC-P1.5-1: CIFAR-10 ≥3α mean ASR ≥ 0.80
    pass_15_1 = cifar_alpha_pass_asr >= 3
    verdicts["AC-P1.5-1"] = (
        "PASS" if pass_15_1 else "FAIL",
        f"CIFAR-10: {cifar_alpha_pass_asr}/4 α configs with mean ASR ≥ 0.80 (need ≥3)"
    )

    # AC-P1.5-2: CIFAR-10 全部 12 组 loss ≠ NaN
    pass_15_2 = cifar_nan_count == 0
    verdicts["AC-P1.5-2"] = (
        "PASS" if pass_15_2 else "FAIL",
        f"CIFAR-10: {cifar_nan_count}/12 experiments with NaN (need 0)"
    )

    # AC-P1.5-3: CIFAR-10 存活实验 clean acc > 30%
    cifar_acc_above_30 = sum(
        1 for alpha in CIFAR_ALPHAS
        for acc in cifar_acc_by_alpha[alpha]
        if acc > 0.30
    )
    cifar_total = sum(len(v) for v in cifar_acc_by_alpha.values())
    pass_15_3 = cifar_acc_above_30 == cifar_total
    verdicts["AC-P1.5-3"] = (
        "PASS" if pass_15_3 else "FAIL",
        f"CIFAR-10: {cifar_acc_above_30}/{cifar_total} experiments with clean acc > 30%"
    )

    # AC-P1.5-4: ΔASR(auto vs γ=1) ≥ 0.30
    if delta_asrs:
        max_delta = max(delta_asrs)
        cifar_deltas = [d for d in delta_asrs[:2]]  # first two are CIFAR
        if cifar_deltas:
            pass_15_4 = max(cifar_deltas) >= 0.30
        else:
            pass_15_4 = False
        verdicts["AC-P1.5-4"] = (
            "PASS" if pass_15_4 else "FAIL",
            f"ΔASR values: {[f'{d:.4f}' for d in delta_asrs]} (need ≥0.30)"
        )
    else:
        verdicts["AC-P1.5-4"] = ("N/A", "No control data available")

    # AC-P1.5-5: MNIST ≥3α median ASR ≥ 0.80
    pass_15_5 = mnist_alpha_pass >= 3
    verdicts["AC-P1.5-5"] = (
        "PASS" if pass_15_5 else "FAIL",
        f"MNIST: {mnist_alpha_pass}/4 α configs with median ASR ≥ 0.80 (need ≥3)"
    )

    # AC-P1.5-6: MNIST α=100 全部 5 seeds ASR > 0.90
    iid_asrs = mnist_asr_by_alpha.get(100, [])
    iid_pass = all(a > 0.90 for a in iid_asrs) if iid_asrs else False
    verdicts["AC-P1.5-6"] = (
        "PASS" if iid_pass else "FAIL",
        f"MNIST α=100 ASRs: {[f'{a:.4f}' for a in iid_asrs]} (all need > 0.90)"
    )

    # AC-P1.5-7: MNIST 全部 loss ≠ NaN
    pass_15_7 = mnist_nan_count == 0
    verdicts["AC-P1.5-7"] = (
        "PASS" if pass_15_7 else "FAIL",
        f"MNIST: {mnist_nan_count}/20 experiments with NaN (need 0)"
    )

    # AC-P1.5-8: 因果性 ΔASR ≥ 0.30 (≥2/3 控制组)
    if delta_asrs:
        ctrl_pass = sum(1 for d in delta_asrs if d >= 0.30)
        pass_15_8 = ctrl_pass >= 2
        verdicts["AC-P1.5-8"] = (
            "PASS" if pass_15_8 else "FAIL",
            f"{ctrl_pass}/3 control groups with ΔASR ≥ 0.30 (need ≥2)"
        )
    else:
        verdicts["AC-P1.5-8"] = ("N/A", "No control data available")

    # Print verdicts
    print()
    all_pass = True
    for ac, (status, detail) in sorted(verdicts.items()):
        icon = {"PASS": "✅", "FAIL": "❌", "CHECK": "🔍", "N/A": "⚪"}.get(status, "?")
        print(f"  {icon} {ac}: {status} — {detail}")
        if status == "FAIL":
            all_pass = False

    print()
    if all_pass:
        print("🎉 ALL AC CRITERIA PASSED (pending manual CHECK items)")
    else:
        print("⚠️  Some AC criteria FAILED — see details above")

    # AC-14, AC-15: Clean accuracy drop (record only)
    print("\n" + "=" * 80)
    print("Record-Only Items (non-blocking)")
    print("=" * 80)

    print("\nAC-14 CIFAR-10 Clean Drop per α:")
    cifar_drop_pass = 0
    for alpha in CIFAR_ALPHAS:
        drops = cifar_drop_by_alpha.get(alpha, [])
        if drops:
            mean_drop = sum(drops) / len(drops)
            status = "PASS" if mean_drop <= 5.0 else "RECORD_ONLY"
            if mean_drop <= 5.0:
                cifar_drop_pass += 1
            print(f"  α={alpha}: mean drop = {mean_drop:.2f} pp → {status}")

    print(f"\n  AC-14 verdict: {cifar_drop_pass}/4 α configs with drop ≤ 5.0pp (need ≥3)")

    print("\nAC-15 MNIST Clean Drop per α:")
    mnist_drop_pass = 0
    for alpha in MNIST_ALPHAS:
        drops = mnist_drop_by_alpha.get(alpha, [])
        if drops:
            mean_drop = sum(drops) / len(drops)
            status = "PASS" if mean_drop <= 3.0 else "RECORD_ONLY"
            if mean_drop <= 3.0:
                mnist_drop_pass += 1
            print(f"  α={alpha}: mean drop = {mean_drop:.2f} pp → {status}")

    print(f"\n  AC-15 verdict: {mnist_drop_pass}/4 α configs with drop ≤ 3.0pp (need ≥3)")


if __name__ == "__main__":
    analyze()
