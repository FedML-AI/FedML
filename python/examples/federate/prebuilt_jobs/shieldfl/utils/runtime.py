import logging
import os
import random

import numpy as np
import torch

_GPU_RUNTIME_MODES = ("single-gpu-deterministic", "single-gpu-fast", "multi-gpu-throughput")


def configure_runtime(args):
    runtime_mode = getattr(args, "runtime_mode", "cpu-deterministic")
    enforce_determinism = bool(getattr(args, "enforce_determinism", True))
    seed = int(getattr(args, "random_seed", 0))

    # GPU 环境预检：GPU 运行模式要求 CUDA 可用
    if runtime_mode in _GPU_RUNTIME_MODES:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"runtime_mode='{runtime_mode}' requires CUDA, but torch.cuda.is_available() is False. "
                "Switch to 'cpu-deterministic' or ensure a GPU is available."
            )

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if runtime_mode in ("cpu-deterministic", "single-gpu-deterministic"):
        if enforce_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                logging.warning("Deterministic algorithms not fully enabled: %s", exc)

    if runtime_mode == "single-gpu-deterministic":
        # PyTorch >= 1.8 要求此环境变量以允许确定性 CUBLAS 操作
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    elif runtime_mode == "single-gpu-fast":
        # GPU 高吞吐模式：关闭确定性以换取速度
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    setattr(args, "runtime_mode", runtime_mode)
    setattr(args, "sort_client_updates", bool(getattr(args, "sort_client_updates", True)))
    _print_runtime_summary(args, runtime_mode, seed)


def _print_runtime_summary(args, runtime_mode, seed):
    """启动时打印运行时上下文摘要，便于结果追溯。"""
    logging.info(
        "ShieldFL runtime configured: mode=%s seed=%s using_gpu=%s model=%s dataset=%s",
        runtime_mode,
        seed,
        getattr(args, "using_gpu", False),
        getattr(args, "model", "unknown"),
        getattr(args, "dataset", "unknown"),
    )
    # 硬件上下文披露（满足学术需求 §3.3）
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda or "unknown"
        logging.info(
            "GPU hardware context: count=%d name=%s vram=%.1fGB cuda=%s",
            gpu_count,
            gpu_name,
            vram_gb,
            cuda_version,
        )
    else:
        logging.info("GPU hardware context: no CUDA device available, running on CPU")
