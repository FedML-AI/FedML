import copy
import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fedml.core import ServerAggregator
from fedml.core.security.fedml_attacker import FedMLAttacker

from .gpu_accelerator import GPUAccelerator
from .micro_ga_base import MicroGABase
from eval.asr import evaluate_asr, _normalize_trigger
from eval.metrics import MetricsCollector


def aggregate_weighted(weights_results, alpha):
    if len(weights_results) == 0:
        return []
    aggregated = []
    max_idx = int(np.argmax(alpha)) if len(alpha) > 0 else 0
    for layer_idx in range(len(weights_results[0])):
        layer0 = weights_results[0][layer_idx]
        is_float_like = np.issubdtype(layer0.dtype, np.floating) or np.issubdtype(
            layer0.dtype, np.complexfloating
        )
        if not is_float_like:
            aggregated.append(np.array(weights_results[max_idx][layer_idx], copy=True))
            continue
        layer_sum = np.zeros_like(layer0)
        for client_idx in range(len(weights_results)):
            if alpha[client_idx] < 1e-6:
                continue
            layer_sum += float(alpha[client_idx]) * weights_results[client_idx][layer_idx]
        aggregated.append(layer_sum)
    return aggregated


class VeriFLAggregator(ServerAggregator, MicroGABase):
    def __init__(self, model, args, data_assets, device):
        ServerAggregator.__init__(self, model, args)
        MicroGABase.__init__(
            self,
            pop_size=getattr(args, "pop_size", 15),
            generations=getattr(args, "generations", 10),
            lambda_reg=getattr(args, "lambda_reg", 0.1),
            seed=int(getattr(args, "random_seed", 0)),
        )
        self.server_momentum = float(getattr(args, "server_momentum", 0.9))
        self.server_lr = float(getattr(args, "server_lr", 0.3))
        self.global_model_buffer = None
        self.velocity_buffer = None
        self.data_assets = data_assets
        self.state_keys = list(self.model.state_dict().keys())
        self.device = device
        self.gpu_accelerator = GPUAccelerator(
            self.model,
            (data_assets.val_images, data_assets.val_labels),
            device=self.device,
            seed=int(getattr(args, "random_seed", 0)),
        )
        self._last_agg_time: Optional[float] = None

        # 结构化指标采集
        metrics_dir = str(getattr(args, "metrics_output_dir", "./results"))
        setattr(args, "aggregator_type", "verifl")
        self._metrics_collector: Optional[MetricsCollector] = MetricsCollector(metrics_dir, args)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def _ordered_dict_to_ndarrays(self, state_dict: OrderedDict) -> List[np.ndarray]:
        return [value.detach().cpu().numpy() for value in state_dict.values()]

    def _ndarrays_to_ordered_dict(self, arrays: List[np.ndarray]) -> OrderedDict:
        target_state = self.model.state_dict()
        converted = OrderedDict()
        for key, array, target in zip(target_state.keys(), arrays, target_state.values()):
            tensor = torch.as_tensor(array)
            if tensor.dtype != target.dtype:
                tensor = tensor.to(dtype=target.dtype)
            converted[key] = tensor
        return converted

    def on_before_aggregation(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        raw_client_model_or_grad_list = list(raw_client_model_or_grad_list)
        # Phase 2: 接入 FedMLAttacker 模型攻击钩子（byzantine / model_replacement）
        if FedMLAttacker.get_instance().is_model_attack():
            raw_client_model_or_grad_list = FedMLAttacker.get_instance().attack_model(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )
            logging.info(
                "VeriFL on_before_aggregation: FedMLAttacker model attack applied | attack_type=%s",
                FedMLAttacker.get_instance().get_attack_types(),
            )
        client_idxs = [idx for idx in range(len(raw_client_model_or_grad_list))]
        logging.info(
            "VeriFL on_before_aggregation: %s client updates | deterministic_order=%s | order_source=fedml_client_index_iteration",
            len(client_idxs),
            getattr(self.args, "sort_client_updates", True),
        )
        return raw_client_model_or_grad_list, client_idxs

    def calculate_fitness(self, weights_results, alpha=None) -> float:
        if alpha is None:
            alpha = weights_results
            weights_results = None
        if self.gpu_accelerator is not None:
            try:
                loss, model_norm = self.gpu_accelerator.calculate_fitness(alpha)
                if not np.isfinite(loss) or not np.isfinite(model_norm):
                    return 0.0
                cost = loss + self.lambda_reg * model_norm
                return 1.0 / (cost + 1e-12)
            except Exception as exc:
                logging.warning("GPUAccelerator fitness fallback triggered: %s", exc)
        if weights_results is None:
            return 0.0
        try:
            aggregated_params = aggregate_weighted(weights_results, alpha)
            val_loss, _ = self._evaluate_arrays(aggregated_params, self.data_assets.val_loader)
            model_norm = 0.0
            for layer in aggregated_params:
                model_norm += np.sum(layer ** 2)
            model_norm = float(np.sqrt(model_norm))
            if not np.isfinite(val_loss):
                return 0.0
            return 1.0 / (val_loss + self.lambda_reg * model_norm + 1e-12)
        except Exception as exc:
            logging.warning("CPU fitness evaluation failed: %s", exc)
            return 0.0

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        _t0 = time.time()
        weights_results = [
            self._ordered_dict_to_ndarrays(client_state)
            for _, client_state in raw_client_model_or_grad_list
        ]
        if self.gpu_accelerator is not None:
            self.gpu_accelerator.set_client_parameters(weights_results)
        num_clients = len(weights_results)
        population = self._init_population(num_clients)
        best_weights = None
        best_fitness = -float("inf")
        for generation in range(self.generations):
            scores = []
            for alpha in population:
                fitness = self.calculate_fitness(weights_results, alpha)
                scores.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = copy.deepcopy(alpha)
            scores = np.array(scores)
            if best_weights is None or best_fitness <= 0:
                population = self._init_population(num_clients)
                continue
            new_population = [best_weights]
            parents = self._tournament_selection(population, scores)
            idx = 0
            while len(new_population) < self.pop_size:
                parent1 = parents[idx % len(parents)]
                parent2 = parents[(idx + 1) % len(parents)]
                idx += 1
                child = self._mutation(self._crossover(parent1, parent2))
                new_population.append(child)
            population = np.array(new_population[: self.pop_size])
            logging.info(
                "GA generation %s/%s | best_fitness=%.6f",
                generation + 1,
                self.generations,
                best_fitness,
            )
        if best_weights is None:
            best_weights = np.ones(num_clients) / num_clients

        anchor_idx = int(np.argmax(best_weights))
        trainable_mask = self.gpu_accelerator.trainable_mask if self.gpu_accelerator else None
        logging.info("VeriFL phase-1 complete | ga_search best anchor candidate=%s", anchor_idx)

        def calc_l2_norm(params):
            if trainable_mask is None:
                return np.sqrt(sum(np.sum(layer ** 2) for layer in params))
            accum = 0.0
            for layer, is_trainable in zip(params, trainable_mask):
                if is_trainable:
                    accum += float(np.sum(layer ** 2))
            return np.sqrt(accum)

        anchor_norm = calc_l2_norm(weights_results[anchor_idx])
        projected_weights = []
        for client_idx in range(num_clients):
            client_norm = calc_l2_norm(weights_results[client_idx])
            scale = anchor_norm / (client_norm + 1e-9)
            if scale < 0.1:
                logging.warning(
                    "Client %s scaling factor %.6f indicates severe scaling attack pattern",
                    client_idx,
                    scale,
                )
            if trainable_mask is None:
                projected = [layer * scale for layer in weights_results[client_idx]]
            else:
                projected = [
                    (layer * scale if is_trainable else layer)
                    for layer, is_trainable in zip(weights_results[client_idx], trainable_mask)
                ]
            projected_weights.append(projected)
        logging.info("VeriFL phase-2 complete | anchor_projection anchor=%s", anchor_idx)

        ga_aggregated_params = aggregate_weighted(projected_weights, best_weights)
        final_params = []
        if self.global_model_buffer is None:
            final_params = ga_aggregated_params
            self.global_model_buffer = copy.deepcopy(final_params)
            self.velocity_buffer = [np.zeros_like(param) for param in final_params]
            logging.info("VeriFL phase-3 init | server_momentum bootstrap from first global state")
        else:
            new_velocity = []
            for old_global, ga_param, old_velocity in zip(
                self.global_model_buffer, ga_aggregated_params, self.velocity_buffer
            ):
                delta = ga_param - old_global
                velocity = self.server_momentum * old_velocity + delta
                updated = old_global + self.server_lr * velocity
                new_velocity.append(velocity)
                final_params.append(updated)
            self.global_model_buffer = copy.deepcopy(final_params)
            self.velocity_buffer = new_velocity
            logging.info("VeriFL phase-3 complete | server_momentum lr=%.4f momentum=%.4f", self.server_lr, self.server_momentum)
        if self.gpu_accelerator is not None:
            final_params = self.gpu_accelerator.recalibrate_batchnorm(final_params)
            logging.info(
                "VeriFL bn_recalibration complete | enabled=%s",
                self.gpu_accelerator.has_batchnorm,
            )

        logging.info(
            "VeriFL aggregate complete | anchor=%s | best_fitness=%.6f | weights=%s",
            anchor_idx,
            best_fitness,
            np.round(best_weights, 4),
        )
        self._last_agg_time = time.time() - _t0
        logging.info("VeriFL aggregate_time=%.4fs", self._last_agg_time)

        # 状态诊断日志（可通过 debug_state_tracking: true 开启）
        if bool(getattr(self.args, "debug_state_tracking", False)):
            gb_norm = np.sqrt(sum(np.sum(x ** 2) for x in self.global_model_buffer))
            vb_norm = np.sqrt(sum(np.sum(x ** 2) for x in self.velocity_buffer))
            logging.info(
                "VeriFL state_check | global_buffer_norm=%.6f | velocity_buffer_norm=%.6f",
                gb_norm,
                vb_norm,
            )

        return self._ndarrays_to_ordered_dict(final_params)

    def _evaluate_arrays(self, weights: List[np.ndarray], loader):
        model = copy.deepcopy(self.model)
        state_dict = self._ndarrays_to_ordered_dict(weights)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = model(images)
                loss = criterion(logits, labels)
                _, predicted = torch.max(logits, 1)
                total_loss += loss.item() * labels.size(0)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
        average_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return average_loss, accuracy

    def test(self, test_data, device, args):
        self.device = device
        self.gpu_accelerator.device = device
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
            "Server test | loss=%.6f | accuracy=%.4f | samples=%s",
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
