#!/usr/bin/env python3
"""
汇总 ShieldFL 实验结果。

从 results/*.jsonl 中提取最终轮 MA/Loss/ASR/agg_time，
按 (model, dataset, aggregator, attack, defense, alpha) 分组，
在 seeds 上计算 mean ± std，输出 Markdown 表格。

用法：
  python scripts/summarize_results.py [--results_dir ./results] [--format markdown|csv]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict


def load_final_round(path):
    """读取 JSONL 文件，返回最后一行记录。"""
    last = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def parse_filename(basename):
    """
    从文件名解析实验元信息。
    格式: metrics_{model}_{dataset}_{aggregator}_atk{attack}_def{defense}_a{alpha}_seed{seed}.jsonl
    """
    name = basename.replace("metrics_", "").replace(".jsonl", "")
    parts = name.split("_")
    info = {}
    for p in parts:
        if p.startswith("seed"):
            info["seed"] = int(p[4:])
        elif p.startswith("atk"):
            info["attack"] = p[3:]
        elif p.startswith("def"):
            info["defense"] = p[3:]
        elif p.startswith("a") and p[1:].replace(".", "", 1).isdigit():
            info["alpha"] = p[1:]
    return info


def extract_alpha_from_record(record):
    """从记录中获取 alpha（如果有的话）。"""
    # alpha 不在标准 metrics 输出中，需要从文件名或额外字段提取
    return record.get("alpha", None)


def main():
    parser = argparse.ArgumentParser(description="Summarize ShieldFL experiment results")
    parser.add_argument("--results_dir", default="./results", help="Directory with JSONL files")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "metrics_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No result files found in %s" % args.results_dir, file=sys.stderr)
        sys.exit(1)

    # 分组: key = (model, dataset, aggregator, attack, defense, alpha)
    groups = defaultdict(list)

    for filepath in files:
        basename = os.path.basename(filepath)
        record = load_final_round(filepath)
        if record is None:
            if args.verbose:
                print("SKIP (empty): %s" % basename, file=sys.stderr)
            continue

        file_info = parse_filename(basename)
        model = record.get("model", "?")
        dataset = record.get("dataset", "?")
        aggregator = record.get("aggregator", "?")
        attack = record.get("attack_type", file_info.get("attack", "?"))
        defense = record.get("defense_type", file_info.get("defense", "?"))
        alpha = record.get("alpha", "?")
        seed = file_info.get("seed", "?")

        key = (model, dataset, aggregator, attack, defense, str(alpha))
        groups[key].append({
            "seed": seed,
            "ma": record.get("test_accuracy", None),
            "loss": record.get("test_loss", None),
            "asr": record.get("asr", None),
            "agg_time": record.get("agg_time", None),
            "rounds": record.get("round", None),
            "runtime_mode": record.get("runtime_mode", "?"),
            "device": record.get("device", "?"),
            "filepath": filepath,
        })

    if not groups:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    # 计算统计
    import math

    def mean_std(values):
        values = [v for v in values if v is not None]
        if not values:
            return None, None
        m = sum(values) / len(values)
        if len(values) > 1:
            s = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
        else:
            s = 0.0
        return m, s

    rows = []
    for key in sorted(groups.keys()):
        model, dataset, aggregator, attack, defense, alpha = key
        entries = groups[key]
        n_seeds = len(entries)

        ma_mean, ma_std = mean_std([e["ma"] for e in entries])
        loss_mean, loss_std = mean_std([e["loss"] for e in entries])
        asr_mean, asr_std = mean_std([e["asr"] for e in entries])
        agg_mean, _ = mean_std([e["agg_time"] for e in entries])
        max_round = max((e["rounds"] for e in entries if e["rounds"] is not None), default=None)
        device = entries[0]["device"]
        runtime = entries[0]["runtime_mode"]

        rows.append({
            "model": model,
            "dataset": dataset,
            "aggregator": aggregator,
            "attack": attack,
            "defense": defense,
            "alpha": alpha,
            "n_seeds": n_seeds,
            "ma_mean": ma_mean,
            "ma_std": ma_std,
            "loss_mean": loss_mean,
            "asr_mean": asr_mean,
            "asr_std": asr_std,
            "agg_time": agg_mean,
            "rounds": max_round,
            "device": device,
            "runtime": runtime,
        })

    def fmt_pct(val, std=None):
        if val is None:
            return "N/A"
        s = "%.2f%%" % (val * 100)
        if std is not None:
            s += " ± %.2f%%" % (std * 100)
        return s

    def fmt_loss(val, std=None):
        if val is None:
            return "N/A"
        s = "%.4f" % val
        if std is not None:
            s += " ± %.4f" % std
        return s

    def fmt_time(val):
        if val is None:
            return "N/A"
        return "%.3fs" % val

    if args.format == "markdown":
        header = "| Model | Dataset | Aggregator | Attack | Defense | α | Seeds | MA (mean±std) | Loss | ASR (mean±std) | AggTime | Rounds | Device |"
        sep =    "|-------|---------|------------|--------|---------|---|-------|---------------|------|----------------|---------|--------|--------|"
        print(header)
        print(sep)
        for r in rows:
            print("| %s | %s | %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
                r["model"], r["dataset"], r["aggregator"], r["attack"], r["defense"],
                r["alpha"], r["n_seeds"],
                fmt_pct(r["ma_mean"], r["ma_std"]),
                fmt_loss(r["loss_mean"]),
                fmt_pct(r["asr_mean"], r["asr_std"]),
                fmt_time(r["agg_time"]),
                r["rounds"] if r["rounds"] is not None else "?",
                r["device"],
            ))
    elif args.format == "csv":
        import csv
        writer = csv.writer(sys.stdout)
        writer.writerow(["model", "dataset", "aggregator", "attack", "defense", "alpha",
                         "n_seeds", "ma_mean", "ma_std", "loss_mean", "asr_mean", "asr_std",
                         "agg_time", "rounds", "device", "runtime"])
        for r in rows:
            writer.writerow([
                r["model"], r["dataset"], r["aggregator"], r["attack"], r["defense"],
                r["alpha"], r["n_seeds"],
                r["ma_mean"], r["ma_std"], r["loss_mean"],
                r["asr_mean"], r["asr_std"], r["agg_time"],
                r["rounds"], r["device"], r["runtime"],
            ])

    # 打印摘要
    print("\n--- Summary ---")
    print("Total experiment groups: %d" % len(rows))
    print("Total result files: %d" % len(files))

    # M1 门槛检查
    print("\n--- M1 Threshold Check ---")
    m1_thresholds_cifar = {"100": 0.85, "0.5": 0.82, "0.3": 0.78, "0.1": 0.75}
    m1_threshold_mnist = 0.97
    for r in rows:
        if r["attack"] == "none" and r["defense"] == "none" and r["aggregator"] == "fedavg":
            if r["dataset"] == "cifar10" and r["alpha"] in m1_thresholds_cifar:
                threshold = m1_thresholds_cifar[r["alpha"]]
                passed = r["ma_mean"] is not None and r["ma_mean"] >= threshold
                status = "PASS" if passed else "FAIL"
                print("  %s+%s α=%s: MA=%.4f (threshold=%.2f) [%s]" % (
                    r["model"], r["dataset"], r["alpha"],
                    r["ma_mean"] if r["ma_mean"] else 0, threshold, status))
            elif r["dataset"] == "mnist":
                passed = r["ma_mean"] is not None and r["ma_mean"] >= m1_threshold_mnist
                status = "PASS" if passed else "FAIL"
                print("  %s+%s α=%s: MA=%.4f (threshold=%.2f) [%s]" % (
                    r["model"], r["dataset"], r["alpha"],
                    r["ma_mean"] if r["ma_mean"] else 0, m1_threshold_mnist, status))


if __name__ == "__main__":
    main()
