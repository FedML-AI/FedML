"""
后门攻击成功率 (ASR) 评估。

触发器注入规则：
- 在图片右下角 trigger_size × trigger_size 区域填充 trigger_value
- trigger_value 语义为原始像素空间值 [0, 1]，代码自动转为归一化空间
- 仅对 label != target_label 的样本注入触发器
- ASR = 被模型预测为 target_label 的比例

使用场景：
- model_replacement_backdoor_attack 实验的 ASR 评估
"""
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Normalization parameters — must match data_loader.py exactly.
_NORM_PARAMS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
}


def _normalize_trigger(pixel_value: float, dataset: str) -> torch.Tensor:
    """Convert pixel-space trigger value to per-channel normalized tensor."""
    key = dataset.lower()
    params = _NORM_PARAMS.get(key)
    if params is None:
        return torch.tensor(pixel_value)
    mean, std = params["mean"], params["std"]
    normalized = [(pixel_value - m) / s for m, s in zip(mean, std)]
    return torch.tensor(normalized, dtype=torch.float32).view(-1, 1, 1)


def evaluate_asr(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_label: int = 0,
    trigger_size: int = 3,
    trigger_value: float = 1.0,
    dataset: str = "",
) -> Dict[str, float]:
    """计算后门攻击成功率 (ASR)。

    Args:
        model: 待评估的模型。
        test_loader: 测试数据加载器（干净数据）。
        device: 运行设备。
        target_label: 触发器目标标签（攻击者希望模型预测的类别）。
        trigger_size: 右下角触发器边长（像素）。
        trigger_value: 触发器填充值（原始像素空间 [0, 1]）。
        dataset: 数据集名称（用于自动归一化转换）。

    Returns:
        dict: {"asr": float, "asr_total": int, "asr_success": int}
    """
    trigger = _normalize_trigger(trigger_value, dataset).to(device)
    logging.info(
        "ASR trigger normalization | dataset=%s | pixel_value=%.4f | normalized=%s",
        dataset, trigger_value,
        trigger.flatten().tolist(),
    )

    model.eval()
    total = 0
    success = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.clone().to(device)
            labels = labels.to(device)
            non_target_mask = (labels != target_label)
            if non_target_mask.sum() == 0:
                continue
            images[:, :, -trigger_size:, -trigger_size:] = trigger
            logits = model(images)
            _, predicted = torch.max(logits, dim=1)
            success += ((predicted == target_label) & non_target_mask).sum().item()
            total += non_target_mask.sum().item()

    asr = success / max(1, total)
    logging.info("ASR evaluation | total=%d | success=%d | asr=%.4f", total, success, asr)
    return {"asr": asr, "asr_total": total, "asr_success": success}
