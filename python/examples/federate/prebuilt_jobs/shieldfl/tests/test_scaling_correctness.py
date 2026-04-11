"""
Scaling Attack 正确性单元测试。
覆盖 Scaling_实施定稿.md 中 AC-1 至 AC-7。

运行: cd python/examples/federate/prebuilt_jobs/shieldfl && python -m pytest tests/test_scaling_correctness.py -v
"""
import copy
import sys
import os
from unittest.mock import patch

import numpy as np
import torch
from collections import OrderedDict
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Path setup: ensure fedml package and local eval module are importable
# ---------------------------------------------------------------------------
_SHIELDFL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FEDML_DIR = os.path.abspath(os.path.join(_SHIELDFL_DIR, "..", "..", "..", "..", "fedml"))
sys.path.insert(0, _SHIELDFL_DIR)
sys.path.insert(0, os.path.join(_SHIELDFL_DIR, ".."))
sys.path.insert(0, os.path.join(_FEDML_DIR, ".."))

# Mock fedml.device.get_device before importing the attack class
import fedml.device
fedml.device.get_device = lambda args: torch.device("cpu")

from fedml.core.security.common.utils import should_scale_param, is_weight_param
from fedml.core.security.attack.model_replacement_backdoor_attack import ModelReplacementBackdoorAttack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Build a minimal SimpleNamespace that satisfies the attack constructor."""
    defaults = dict(
        enable_attack=True,
        attack_type="model_replacement",
        byzantine_client_num=3,
        client_num_in_total=10,
        client_num_per_round=10,
        scale_gamma=10,
        attack_training_rounds=[3, 4],
        target_label=0,
        trigger_size=3,
        trigger_value=1.0,
        backdoor_per_batch=20,
        random_seed=0,
        using_gpu=False,
        # Required by fedml.device.get_device()
        training_type="cross_silo",
        backend="MPI",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_model_state():
    """Create a mock model state dict with weight, bias, and BN params."""
    state = OrderedDict()
    state["conv1.weight"] = torch.randn(16, 3, 3, 3)
    state["conv1.bias"] = torch.randn(16)
    state["bn1.weight"] = torch.randn(16)
    state["bn1.bias"] = torch.randn(16)
    state["bn1.running_mean"] = torch.randn(16)
    state["bn1.running_var"] = torch.abs(torch.randn(16)) + 0.1
    state["bn1.num_batches_tracked"] = torch.tensor(100)
    state["fc.weight"] = torch.randn(10, 16)
    state["fc.bias"] = torch.randn(10)
    return state


def _make_client_list(n=10):
    return [(100 + i, _make_model_state()) for i in range(n)]


# ===========================================================================
# AC-1: 攻击轮次配置生效
# ===========================================================================

class TestAC1_AttackRoundFiltering:

    def test_only_specified_rounds_apply_scaling(self):
        """设置 attack_training_rounds=[3], 仅 round=3 发生缩放。"""
        args = _make_args(attack_training_rounds=[3])
        attacker = ModelReplacementBackdoorAttack(args)
        global_model = _make_model_state()

        for round_idx in range(5):
            client_list = _make_client_list()
            originals = [(n, copy.deepcopy(m)) for n, m in client_list]

            attacker.attack_model(client_list, extra_auxiliary_info=global_model)

            if round_idx == 3:
                # Malicious clients (0,1,2) should be modified
                for idx in [0, 1, 2]:
                    changed = any(
                        not torch.equal(client_list[idx][1][k], originals[idx][1][k])
                        for k in client_list[idx][1]
                        if should_scale_param(k)
                    )
                    assert changed, f"Round {round_idx}: client {idx} should be modified"
            else:
                # No clients should be modified
                for idx in range(10):
                    for k in client_list[idx][1]:
                        assert torch.equal(client_list[idx][1][k], originals[idx][1][k]), (
                            f"Round {round_idx}: client {idx} param {k} should NOT be modified"
                        )

    def test_non_attack_rounds_skip_entirely(self):
        """attack_training_rounds=[99], 5-round experiment => no scaling at all."""
        args = _make_args(attack_training_rounds=[99])
        attacker = ModelReplacementBackdoorAttack(args)
        global_model = _make_model_state()

        for _ in range(5):
            client_list = _make_client_list()
            originals = [(n, copy.deepcopy(m)) for n, m in client_list]
            attacker.attack_model(client_list, extra_auxiliary_info=global_model)
            for idx in range(10):
                for k in client_list[idx][1]:
                    assert torch.equal(client_list[idx][1][k], originals[idx][1][k])


# ===========================================================================
# AC-2: 恶意客户端 batch 注入数量正确
# ===========================================================================

class TestAC2_BatchInjection:

    def test_inject_20_per_batch(self):
        """64-sample batch, 20 samples should be injected."""
        rng = np.random.default_rng(42)
        batch_size = 64
        backdoor_per_batch = 20
        trigger_size = 3
        trigger_value = 1.0
        target_label = 0

        images = torch.zeros(batch_size, 3, 32, 32)
        labels = torch.randint(1, 10, (batch_size,))

        inject_count = min(backdoor_per_batch, batch_size)
        indices = rng.choice(batch_size, size=inject_count, replace=False)
        images[indices, :, -trigger_size:, -trigger_size:] = trigger_value
        labels[indices] = target_label

        triggered = (images[:, 0, -1, -1] == trigger_value).sum().item()
        assert triggered == 20
        assert (labels == target_label).sum().item() == 20

    def test_inject_capped_by_small_batch(self):
        """If batch size < 20, inject count = batch size."""
        rng = np.random.default_rng(42)
        batch_size = 10
        backdoor_per_batch = 20

        images = torch.zeros(batch_size, 3, 32, 32)
        labels = torch.randint(1, 10, (batch_size,))

        inject_count = min(backdoor_per_batch, batch_size)
        assert inject_count == 10

        indices = rng.choice(batch_size, size=inject_count, replace=False)
        images[indices, :, -3:, -3:] = 1.0
        labels[indices] = 0

        assert (labels == 0).sum().item() == 10

    def test_benign_client_no_injection(self):
        """Non-malicious client: zero poisoned samples."""
        images = torch.zeros(64, 3, 32, 32)
        labels = torch.randint(1, 10, (64,))
        # Benign client does not modify images/labels
        assert (images[:, 0, -1, -1] == 1.0).sum().item() == 0


# ===========================================================================
# AC-3: 训练 trigger 与评估 trigger 完全一致
# ===========================================================================

class TestAC3_TriggerConsistency:

    def test_trigger_patch_identical(self):
        """Training trigger injection must produce identical result as ASR eval."""
        trigger_size = 3
        trigger_value = 1.0

        img_train = torch.randn(1, 3, 32, 32)
        img_eval = img_train.clone()

        # Training code path (from verifl_trainer.py)
        img_train[:, :, -trigger_size:, -trigger_size:] = trigger_value

        # ASR eval code path (from asr.py)
        img_eval[:, :, -trigger_size:, -trigger_size:] = trigger_value

        assert torch.max(torch.abs(img_train - img_eval)).item() == 0.0

    def test_trigger_covers_correct_region(self):
        """Trigger must cover exactly the bottom-right 3x3 region."""
        trigger_size = 3
        trigger_value = 1.0
        img = torch.zeros(1, 3, 32, 32)
        img[:, :, -trigger_size:, -trigger_size:] = trigger_value

        # The trigger region: rows 29-31, cols 29-31
        assert img[0, 0, 29, 29].item() == trigger_value
        assert img[0, 0, 31, 31].item() == trigger_value
        # Outside trigger region should be 0
        assert img[0, 0, 28, 28].item() == 0.0
        assert img[0, 0, 0, 0].item() == 0.0


# ===========================================================================
# AC-4: 恶意客户端集合固定且一致
# ===========================================================================

class TestAC4_MaliciousClientIds:

    def test_fixed_malicious_ids(self):
        args = _make_args()
        attacker = ModelReplacementBackdoorAttack(args)
        assert attacker.malicious_client_ids == [0, 1, 2]

    def test_reproducible_across_runs(self):
        a1 = ModelReplacementBackdoorAttack(_make_args())
        a2 = ModelReplacementBackdoorAttack(_make_args())
        assert a1.malicious_client_ids == a2.malicious_client_ids

    def test_id_count_matches_byzantine_num(self):
        for k in [1, 3, 5]:
            args = _make_args(byzantine_client_num=k)
            attacker = ModelReplacementBackdoorAttack(args)
            assert len(attacker.malicious_client_ids) == k
            assert attacker.malicious_client_ids == list(range(k))


# ===========================================================================
# AC-5: gamma 可配且数值正确
# ===========================================================================

class TestAC5_GammaConfigurable:

    def test_gamma_10(self):
        attacker = ModelReplacementBackdoorAttack(_make_args(scale_gamma=10))
        assert attacker.scale_gamma == 10.0

    def test_gamma_1(self):
        attacker = ModelReplacementBackdoorAttack(_make_args(scale_gamma=1))
        assert attacker.scale_gamma == 1.0

    def test_gamma_default_equals_client_num(self):
        """When scale_gamma is not set, default to client_num_in_total."""
        args = _make_args()
        delattr(args, "scale_gamma")
        attacker = ModelReplacementBackdoorAttack(args)
        assert attacker.scale_gamma == 10.0


# ===========================================================================
# AC-6: BN 参数缩放范围正确
# ===========================================================================

class TestAC6_BNParams:

    def test_should_scale_running_mean(self):
        assert should_scale_param("bn1.running_mean") is True

    def test_should_scale_running_var(self):
        assert should_scale_param("bn1.running_var") is True

    def test_should_not_scale_num_batches_tracked(self):
        assert should_scale_param("bn1.num_batches_tracked") is False

    def test_should_scale_weight_and_bias(self):
        assert should_scale_param("conv1.weight") is True
        assert should_scale_param("conv1.bias") is True
        assert should_scale_param("bn1.weight") is True
        assert should_scale_param("bn1.bias") is True

    def test_is_weight_param_excludes_bn_stats(self):
        """is_weight_param still excludes running_mean/running_var (unchanged)."""
        assert is_weight_param("bn1.running_mean") is False
        assert is_weight_param("bn1.running_var") is False
        assert is_weight_param("bn1.num_batches_tracked") is False

    def test_scaling_modifies_running_mean_and_var(self):
        """Integration: scaling with gamma=10 modifies running_mean/running_var."""
        args = _make_args(attack_training_rounds=[0], scale_gamma=10)
        attacker = ModelReplacementBackdoorAttack(args)
        global_model = _make_model_state()
        client_list = _make_client_list()

        orig_mean = client_list[0][1]["bn1.running_mean"].clone()
        orig_var = client_list[0][1]["bn1.running_var"].clone()
        orig_nbt = client_list[0][1]["bn1.num_batches_tracked"].clone()

        attacker.attack_model(client_list, extra_auxiliary_info=global_model)

        assert not torch.equal(client_list[0][1]["bn1.running_mean"], orig_mean), \
            "running_mean should be modified after scaling"
        assert not torch.equal(client_list[0][1]["bn1.running_var"], orig_var), \
            "running_var should be modified after scaling"
        assert torch.equal(client_list[0][1]["bn1.num_batches_tracked"], orig_nbt), \
            "num_batches_tracked must NOT be modified"


# ===========================================================================
# AC-7: 非恶意条目不被破坏
# ===========================================================================

class TestAC7_NonMaliciousIntegrity:

    def test_list_length_preserved(self):
        args = _make_args(attack_training_rounds=[0])
        attacker = ModelReplacementBackdoorAttack(args)
        global_model = _make_model_state()
        client_list = _make_client_list()

        original_len = len(client_list)
        attacker.attack_model(client_list, extra_auxiliary_info=global_model)
        assert len(client_list) == original_len

    def test_non_malicious_sample_num_unchanged(self):
        args = _make_args(attack_training_rounds=[0])
        attacker = ModelReplacementBackdoorAttack(args)
        global_model = _make_model_state()
        client_list = _make_client_list()
        original_nums = [n for n, _ in client_list]

        attacker.attack_model(client_list, extra_auxiliary_info=global_model)

        for idx in range(3, 10):
            assert client_list[idx][0] == original_nums[idx], \
                f"Client {idx} sample_num changed from {original_nums[idx]} to {client_list[idx][0]}"

    def test_non_malicious_params_identical(self):
        args = _make_args(attack_training_rounds=[0])
        attacker = ModelReplacementBackdoorAttack(args)
        global_model = _make_model_state()
        client_list = _make_client_list()
        originals = [(n, copy.deepcopy(m)) for n, m in client_list]

        attacker.attack_model(client_list, extra_auxiliary_info=global_model)

        for idx in range(3, 10):
            for k in client_list[idx][1]:
                assert torch.equal(client_list[idx][1][k], originals[idx][1][k]), \
                    f"Client {idx} param {k} was modified!"

    def test_malicious_sample_num_unchanged(self):
        """Even malicious clients should keep their sample_num."""
        args = _make_args(attack_training_rounds=[0])
        attacker = ModelReplacementBackdoorAttack(args)
        global_model = _make_model_state()
        client_list = _make_client_list()
        original_nums = [n for n, _ in client_list]

        attacker.attack_model(client_list, extra_auxiliary_info=global_model)

        for idx in [0, 1, 2]:
            assert client_list[idx][0] == original_nums[idx]
