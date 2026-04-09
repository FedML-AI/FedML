import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from fedml.core import ServerAggregator

from fedml.core.security.fedml_attacker import FedMLAttacker

from eval.asr import evaluate_asr, _normalize_trigger
from eval.metrics import MetricsCollector


class ShieldFLAggregator(ServerAggregator):
    def __init__(self, model, args, data_assets=None, device=None):
        super().__init__(model, args)
        self.data_assets = data_assets
        self.device = device if device is not None else torch.device("cpu")
        self._last_agg_time = None

        metrics_dir = str(getattr(args, "metrics_output_dir", "./results"))
        self._metrics_collector = MetricsCollector(metrics_dir, args)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def test(self, test_data, device, args):
        self.device = device
        model = self.model
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        metrics: Dict[str, float] = {
            "test_correct": 0,
            "test_loss": 0.0,
            "test_total": 0,
            "test_accuracy": 0.0,
        }
        with torch.no_grad():
            for images, labels in test_data:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                _, predicted = torch.max(logits, 1)
                metrics["test_correct"] += predicted.eq(labels).sum().item()
                metrics["test_loss"] += loss.item() * labels.size(0)
                metrics["test_total"] += labels.size(0)
        if metrics["test_total"] > 0:
            metrics["test_accuracy"] = metrics["test_correct"] / metrics["test_total"]
            metrics["test_loss"] = metrics["test_loss"] / metrics["test_total"]
        logging.info(
            "ShieldFLAggregator test | loss=%.6f | accuracy=%.4f | samples=%s",
            metrics["test_loss"],
            metrics["test_accuracy"],
            metrics["test_total"],
        )

        # ASR 评估
        asr_value = None
        if bool(getattr(args, "eval_asr", False)) and self.data_assets is not None:
            asr_result = evaluate_asr(
                model=model,
                test_loader=self.data_assets.test_loader,
                device=device,
                target_label=int(getattr(args, "target_label", 0)),
                trigger_size=int(getattr(args, "trigger_size", 3)),
                trigger_value=float(getattr(args, "trigger_value", 1.0)),
                dataset=str(getattr(args, "dataset", "")),
            )
            asr_value = asr_result["asr"]
            metrics.update({f"asr_{k}": v for k, v in asr_result.items()})

        # 写入结构化指标
        if self._metrics_collector is not None:
            round_idx = int(getattr(args, "round_idx", -1))
            gamma_actual = None
            attacker_inst = FedMLAttacker.get_instance()
            if attacker_inst.is_enabled and attacker_inst.attacker is not None:
                gamma_actual = getattr(attacker_inst.attacker, "last_gamma", None)
            malicious_count = int(getattr(args, "byzantine_client_num", 0)) if bool(getattr(args, "enable_attack", False)) else None
            ds = str(getattr(args, "dataset", ""))
            tv = float(getattr(args, "trigger_value", 1.0))
            trigger_norm = _normalize_trigger(tv, ds).flatten().tolist() if ds else None
            self._metrics_collector.log_round(
                round_idx=round_idx,
                test_accuracy=metrics["test_accuracy"],
                test_loss=metrics["test_loss"],
                test_total=int(metrics["test_total"]),
                asr=asr_value,
                agg_time=self._last_agg_time,
                gamma_actual=gamma_actual,
                malicious_count=malicious_count,
                trigger_value_normalized=trigger_norm,
            )
        return metrics

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        return False
