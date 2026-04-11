#!/usr/bin/env python3
"""
冒烟测试：可插拔防御注册表 + VeriFL→v16 重命名

测试层级：
  T1 - 注册表基础功能（纯 Python，无 GPU）
  T2 - 所有内置防御可实例化（无 GPU）
  T3 - 阶段元数据与旧硬编码行为一致性
  T4 - 自定义防御注册 & 使用
  T5 - 重命名后的 import 可用性
  T6 - 状态隔离验证
  T7 - 端到端短训练（需 GPU + MPI）— 单独脚本

用法：
  python tests/smoke_test_pluggable_defense.py          # T1-T6
  python tests/smoke_test_pluggable_defense.py --verbose # 详细输出
"""
import argparse
import logging
import sys
import traceback
from collections import OrderedDict
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_results = []


def _run(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  [PASS] {name}")
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  [FAIL] {name}: {e}")
        if _verbose:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# T1 — 注册表基础功能
# ---------------------------------------------------------------------------
def test_registry_available_defenses():
    from fedml.core.security.fedml_defender import FedMLDefender
    avail = FedMLDefender.available_defenses()
    assert isinstance(avail, list) and len(avail) >= 16, \
        f"Expected >=16 defenses, got {len(avail)}: {avail}"


def test_registry_unknown_defense_raises():
    from fedml.core.security.fedml_defender import _DEFENSE_REGISTRY
    try:
        _DEFENSE_REGISTRY.get("__nonexistent_defense__")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass  # expected


# ---------------------------------------------------------------------------
# T2 — 所有内置防御可实例化
# ---------------------------------------------------------------------------
def test_all_builtin_defenses_resolve():
    from fedml.core.security.fedml_defender import FedMLDefender, _DEFENSE_REGISTRY
    from fedml.core.security.defense.defense_base import BaseDefenseMethod
    avail = FedMLDefender.available_defenses()
    # Verify every registered defense can be *resolved* (lazy import works)
    # and the resolved class is a BaseDefenseMethod subclass.
    failed = []
    for dtype in avail:
        try:
            entry = _DEFENSE_REGISTRY.get(dtype)
            assert entry.cls is not None, "cls is None"
            assert issubclass(entry.cls, BaseDefenseMethod), \
                f"{entry.cls} is not a BaseDefenseMethod subclass"
        except Exception as e:
            failed.append((dtype, str(e)))
    if failed:
        msg = "; ".join(f"{d}: {e}" for d, e in failed)
        raise AssertionError(f"Failed to resolve: {msg}")


# ---------------------------------------------------------------------------
# T3 — 阶段元数据一致性（对比旧硬编码逻辑）
# ---------------------------------------------------------------------------
def test_phase_metadata_consistency():
    """Verify that the registry phases produce the same is_defense_* results
    as the old hardcoded lists."""
    from fedml.core.security.fedml_defender import (
        FedMLDefender, _DEFENSE_REGISTRY,
        PHASE_BEFORE, PHASE_ON, PHASE_AFTER,
    )
    from fedml.core.security.constants import (
        DEFENSE_NORM_DIFF_CLIPPING, DEFENSE_ROBUST_LEARNING_RATE,
        DEFENSE_KRUM, DEFENSE_SLSGD, DEFENSE_GEO_MEDIAN, DEFENSE_CCLIP,
        DEFENSE_WEAK_DP, DEFENSE_RFA, DEFENSE_FOOLSGOLD,
        DEFENSE_THREESIGMA_FOOLSGOLD, DEFENSE_CRFL, DEFENSE_MULTIKRUM,
        DEFENSE_TRIMMED_MEAN, DEFENSE_THREESIGMA_GEOMEDIAN,
        DEFENSE_THREESIGMA, ANOMALY_DETECTION, DEFENSE_WISE_MEDIAN,
    )

    # Old hardcoded sets (from the original code)
    OLD_ON = {DEFENSE_SLSGD, DEFENSE_RFA, DEFENSE_WISE_MEDIAN, DEFENSE_GEO_MEDIAN}
    OLD_BEFORE = {
        DEFENSE_SLSGD, DEFENSE_FOOLSGOLD, DEFENSE_THREESIGMA_FOOLSGOLD,
        DEFENSE_THREESIGMA_GEOMEDIAN, DEFENSE_THREESIGMA, DEFENSE_KRUM,
        DEFENSE_CCLIP, DEFENSE_MULTIKRUM, DEFENSE_TRIMMED_MEAN,
        ANOMALY_DETECTION, DEFENSE_NORM_DIFF_CLIPPING,
    }
    OLD_AFTER = {DEFENSE_CRFL, DEFENSE_CCLIP}

    mismatches = []
    for dtype in FedMLDefender.available_defenses():
        entry = _DEFENSE_REGISTRY.get(dtype)
        new_before = PHASE_BEFORE in entry.phases
        new_on = PHASE_ON in entry.phases
        new_after = PHASE_AFTER in entry.phases
        old_before = dtype in OLD_BEFORE
        old_on = dtype in OLD_ON
        old_after = dtype in OLD_AFTER
        if new_before != old_before:
            mismatches.append(f"{dtype}: before new={new_before} old={old_before}")
        if new_on != old_on:
            mismatches.append(f"{dtype}: on new={new_on} old={old_on}")
        if new_after != old_after:
            mismatches.append(f"{dtype}: after new={new_after} old={old_after}")

    if mismatches:
        raise AssertionError("Phase mismatches:\n  " + "\n  ".join(mismatches))


# ---------------------------------------------------------------------------
# T4 — 自定义防御注册 & 使用
# ---------------------------------------------------------------------------
def test_custom_defense_registration():
    from fedml.core.security.fedml_defender import FedMLDefender, PHASE_BEFORE
    from fedml.core.security.defense.defense_base import BaseDefenseMethod

    class _TestDefense(BaseDefenseMethod):
        def __init__(self, config):
            self._cfg = config
            self.called = False

        def defend_before_aggregation(self, raw_client_grad_list, extra_auxiliary_info=None):
            self.called = True
            return raw_client_grad_list

    FedMLDefender.register_defense("__smoke_test_custom__", _TestDefense, {PHASE_BEFORE})
    assert "__smoke_test_custom__" in FedMLDefender.available_defenses()

    # Verify it can be instantiated via the normal init() path
    args = SimpleNamespace(
        enable_defense=True,
        defense_type="__smoke_test_custom__",
    )
    defender = FedMLDefender.get_instance()
    defender.init(args)
    assert defender.is_enabled
    assert defender.is_defense_before_aggregation()
    assert not defender.is_defense_on_aggregation()
    assert not defender.is_defense_after_aggregation()

    # Verify defend_before_aggregation works
    dummy_grads = [(1.0, OrderedDict({"w": 1.0}))]
    result = defender.defend_before_aggregation(dummy_grads)
    assert result == dummy_grads
    assert defender.defender.called

    # Clean up: disable
    clean_args = SimpleNamespace(enable_defense=False)
    defender.init(clean_args)


# ---------------------------------------------------------------------------
# T5 — 重命名后 import 可用性
# ---------------------------------------------------------------------------
def test_renamed_imports():
    """Verify the renamed modules and classes can be imported."""
    import os, sys
    shieldfl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if shieldfl_dir not in sys.path:
        sys.path.insert(0, shieldfl_dir)
    # These imports prove the file rename + class rename are consistent
    from trainer.verifl_v16_aggregator import VeriFLv16Aggregator
    from trainer.verifl_v16_trainer import VeriFLv16Trainer
    from trainer import VeriFLv16Trainer as T2, ShieldFLAggregator

    assert VeriFLv16Aggregator is not None
    assert VeriFLv16Trainer is T2
    assert ShieldFLAggregator is not None

    # Verify old names do NOT exist
    try:
        from trainer.verifl_aggregator import VeriFLAggregator  # noqa
        raise AssertionError("Old module 'verifl_aggregator' should not exist")
    except ImportError:
        pass

    try:
        from trainer.verifl_trainer import VeriFLTrainer  # noqa
        raise AssertionError("Old module 'verifl_trainer' should not exist")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# T6 — 状态隔离
# ---------------------------------------------------------------------------
def test_state_isolation():
    """Verify that calling init() again fully resets defender state."""
    from fedml.core.security.fedml_defender import FedMLDefender

    defender = FedMLDefender.get_instance()

    # Init with krum
    args1 = SimpleNamespace(
        enable_defense=True,
        defense_type="krum",
        client_num_per_round=10,
        byzantine_client_num=2,
    )
    defender.init(args1)
    assert defender.is_enabled
    assert defender.defense_type == "krum"
    krum_instance = defender.defender
    assert krum_instance is not None

    # Init with defense disabled
    args2 = SimpleNamespace(enable_defense=False)
    defender.init(args2)
    assert not defender.is_enabled
    assert defender.defender is None, f"defender should be None, got {defender.defender}"
    assert defender.defense_type is None, f"defense_type should be None, got {defender.defense_type}"

    # Init with a different defense
    args3 = SimpleNamespace(
        enable_defense=True,
        defense_type="foolsgold",
        client_num_per_round=10,
        byzantine_client_num=2,
    )
    defender.init(args3)
    assert defender.is_enabled
    assert defender.defense_type == "foolsgold"
    assert defender.defender is not krum_instance  # Different instance
    assert type(defender.defender).__name__ != type(krum_instance).__name__

    # Clean up
    defender.init(SimpleNamespace(enable_defense=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parsed = parser.parse_args()
    _verbose = parsed.verbose

    if _verbose:
        logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("ShieldFL Smoke Tests: Pluggable Defense + v16 Rename")
    print("=" * 60)

    print("\n--- T1: Registry basics ---")
    _run("T1.1 available_defenses", test_registry_available_defenses)
    _run("T1.2 unknown_defense_raises", test_registry_unknown_defense_raises)

    print("\n--- T2: Built-in defense resolution ---")
    _run("T2.1 all_builtin_defenses_resolve", test_all_builtin_defenses_resolve)

    print("\n--- T3: Phase metadata consistency ---")
    _run("T3.1 phase_metadata_vs_old_hardcoded", test_phase_metadata_consistency)

    print("\n--- T4: Custom defense registration ---")
    _run("T4.1 register_and_use_custom_defense", test_custom_defense_registration)

    print("\n--- T5: Renamed imports ---")
    _run("T5.1 renamed_modules_importable", test_renamed_imports)

    print("\n--- T6: State isolation ---")
    _run("T6.1 init_resets_state_fully", test_state_isolation)

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("\nFailed tests:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED")
        sys.exit(0)
