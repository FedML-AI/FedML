"""
结构化指标采集器。
每轮 append 一行 JSON 到指标文件，字段包括：
{
    "round": int,
    "aggregator": str,
    "attack_type": str,
    "defense_type": str,
    "test_accuracy": float,
    "test_loss": float,
    "test_total": int,
    "asr": float | null,
    "agg_time": float | null,
    "timestamp": str
}
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional


class MetricsCollector:
    def __init__(self, output_dir: str, args):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        aggregator_type = str(getattr(args, "aggregator_type", "verifl"))
        attack_type = str(getattr(args, "attack_type", "none"))
        defense_type = str(getattr(args, "defense_type", "none"))
        model_name = str(getattr(args, "model", "unknown"))
        dataset = str(getattr(args, "dataset", "unknown"))
        seed = str(getattr(args, "random_seed", 0))
        alpha = str(getattr(args, "partition_alpha", "unknown"))
        pmr = str(getattr(args, "ratio_of_poisoned_client", getattr(args, "pmr", 0.0)))

        # Include scale_gamma in filename to avoid collisions between
        # auto-γ experiments and γ=1 control groups (same α/seed).
        raw_gamma = getattr(args, "scale_gamma", "auto")
        gamma_tag = str(raw_gamma).replace(".", "p")  # e.g. "auto" or "1" or "1p0"
        fname = f"metrics_{model_name}_{dataset}_{aggregator_type}_atk{attack_type}_def{defense_type}_a{alpha}_pmr{pmr}_g{gamma_tag}_seed{seed}.jsonl"
        self.metrics_file = os.path.join(output_dir, fname)
        import torch
        import subprocess
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
        except Exception:
            git_commit = "unknown"
        self._meta = {
            "aggregator": aggregator_type,
            "attack_type": attack_type,
            "defense_type": defense_type,
            "model": model_name,
            "dataset": dataset,
            "alpha": alpha,
            "runtime_mode": str(getattr(args, "runtime_mode", "unknown")),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "comm_round": int(getattr(args, "comm_round", -1)),
            "epochs": int(getattr(args, "epochs", -1)),
            "learning_rate": float(getattr(args, "learning_rate", -1)),
            "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            "batch_size": int(getattr(args, "batch_size", -1)),
            "pmr": float(getattr(args, "ratio_of_poisoned_client", getattr(args, "pmr", 0.0))),
            "random_seed": int(getattr(args, "random_seed", 0)),
            "client_num_in_total": int(getattr(args, "client_num_in_total", -1)),
            "client_num_per_round": int(getattr(args, "client_num_per_round", -1)),
            "momentum": float(getattr(args, "momentum", 0.0)),
            "git_commit": git_commit,
        }
        logging.info("MetricsCollector initialized | output=%s", self.metrics_file)

        # D-14: running maximum ASR across rounds
        self._max_asr: float = 0.0

    def log_round(
        self,
        round_idx: int,
        test_accuracy: float,
        test_loss: float,
        test_total: int,
        asr: Optional[float] = None,
        agg_time: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        gamma_actual: Optional[float] = None,
        malicious_count: Optional[int] = None,
        trigger_value_normalized: Optional[list] = None,
    ):
        # D-14: update running max ASR
        if asr is not None:
            self._max_asr = max(self._max_asr, float(asr))

        record = {
            "round": round_idx,
            "test_accuracy": round(float(test_accuracy), 6),
            "test_loss": round(float(test_loss), 6),
            "test_total": int(test_total),
            "asr": round(float(asr), 6) if asr is not None else None,
            "max_asr": round(self._max_asr, 6) if asr is not None else None,
            "gamma_actual": round(float(gamma_actual), 4) if gamma_actual is not None else None,
            "malicious_count": int(malicious_count) if malicious_count is not None else None,
            "trigger_value_normalized": trigger_value_normalized,
            "agg_time": round(float(agg_time), 4) if agg_time is not None else None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        record.update(self._meta)
        if extra:
            record.update(extra)
        try:
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logging.warning("MetricsCollector failed to write: %s", exc)
