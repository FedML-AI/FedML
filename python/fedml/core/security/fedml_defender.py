import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Type, Any, Callable, Set

from ..common.ml_engine_backend import MLEngineBackend
from .defense.defense_base import BaseDefenseMethod
from ...core.security.constants import (
    DEFENSE_NORM_DIFF_CLIPPING,
    DEFENSE_ROBUST_LEARNING_RATE,
    DEFENSE_KRUM,
    DEFENSE_SLSGD,
    DEFENSE_GEO_MEDIAN,
    DEFENSE_CCLIP,
    DEFENSE_WEAK_DP,
    DEFENSE_RFA,
    DEFENSE_FOOLSGOLD,
    DEFENSE_THREESIGMA_FOOLSGOLD,
    DEFENSE_CRFL,
    DEFENSE_MULTIKRUM,
    DEFENSE_TRIMMED_MEAN,
    DEFENSE_THREESIGMA_GEOMEDIAN,
    DEFENSE_THREESIGMA, ANOMALY_DETECTION, DEFENSE_WISE_MEDIAN, DEFENSE_DIFF_CLIPPING,
)


# ---------------------------------------------------------------------------
# Defense hook-phase declarations
# ---------------------------------------------------------------------------
# Each defense class is registered together with the set of hook phases it
# participates in.  The dispatcher uses this metadata instead of hardcoded
# lists so that a newly registered defense automatically works without
# touching any code inside FedMLDefender.
PHASE_BEFORE = "before_aggregation"
PHASE_ON = "on_aggregation"
PHASE_AFTER = "after_aggregation"


class _DefenseEntry:
    """Internal descriptor stored in the registry for each defense type."""
    __slots__ = ("cls", "phases", "aliases")

    def __init__(self, cls: Type[BaseDefenseMethod], phases: Set[str],
                 aliases: Tuple[str, ...] = ()):
        self.cls = cls
        self.phases = frozenset(phases)
        self.aliases = aliases


def _lazy_import(module_path: str, class_name: str):
    """Return a callable that imports *class_name* from *module_path* on first call."""
    _cache = {}

    def _resolve():
        if "cls" not in _cache:
            import importlib
            mod = importlib.import_module(module_path, package=__package__)
            _cache["cls"] = getattr(mod, class_name)
        return _cache["cls"]

    return _resolve


class _DefenseRegistry:
    """
    Central, **class-level** registry that maps defense_type strings to their
    implementation classes and hook-phase metadata.

    Design goals:
    - **Open for extension**: external code can register new defenses via
      ``FedMLDefender.register_defense()`` without modifying this file.
    - **State isolation**: every ``init()`` call creates a *fresh* defense
      instance — no cross-experiment state leakage.
    - **Lazy imports**: built-in defense modules are only imported when
      actually requested, keeping startup fast.
    """

    def __init__(self):
        # {defense_type_str: _DefenseEntry}
        self._entries: Dict[str, _DefenseEntry] = {}
        # Deferred entries use a resolver callable instead of an actual class.
        # They are resolved on first lookup.
        self._deferred: Dict[str, Tuple[Callable, Set[str]]] = {}

    # -- registration helpers ------------------------------------------------

    def register(self, defense_type: str, cls: Type[BaseDefenseMethod],
                 phases: Set[str], aliases: Tuple[str, ...] = ()):
        entry = _DefenseEntry(cls, phases, aliases)
        self._entries[defense_type] = entry
        for alias in aliases:
            self._entries[alias] = entry

    def register_lazy(self, defense_type: str, module_path: str,
                      class_name: str, phases: Set[str],
                      aliases: Tuple[str, ...] = ()):
        resolver = _lazy_import(module_path, class_name)
        self._deferred[defense_type] = (resolver, phases)
        for alias in aliases:
            self._deferred[alias] = (resolver, phases)

    # -- lookup --------------------------------------------------------------

    def get(self, defense_type: str) -> _DefenseEntry:
        if defense_type in self._entries:
            return self._entries[defense_type]
        if defense_type in self._deferred:
            resolver, phases = self._deferred.pop(defense_type)
            cls = resolver()
            entry = _DefenseEntry(cls, phases)
            self._entries[defense_type] = entry
            return entry
        raise ValueError(
            f"Unknown defense_type '{defense_type}'. "
            f"Available: {sorted(set(list(self._entries.keys()) + list(self._deferred.keys())))}"
        )

    def available(self):
        return sorted(set(list(self._entries.keys()) + list(self._deferred.keys())))


# ---------------------------------------------------------------------------
# Global registry instance — populated once at import time
# ---------------------------------------------------------------------------
_DEFENSE_REGISTRY = _DefenseRegistry()

# Built-in defenses (lazy-imported so we don't pay startup cost for all 17+)
_BUILTIN = [
    (DEFENSE_NORM_DIFF_CLIPPING, ".defense.norm_diff_clipping_defense", "NormDiffClippingDefense",
     {PHASE_BEFORE}, ()),
    (DEFENSE_ROBUST_LEARNING_RATE, ".defense.robust_learning_rate_defense", "RobustLearningRateDefense",
     set(), ()),
    (DEFENSE_KRUM, ".defense.krum_defense", "KrumDefense",
     {PHASE_BEFORE}, (DEFENSE_MULTIKRUM,)),
    (DEFENSE_SLSGD, ".defense.slsgd_defense", "SLSGDDefense",
     {PHASE_BEFORE, PHASE_ON}, ()),
    (DEFENSE_GEO_MEDIAN, ".defense.geometric_median_defense", "GeometricMedianDefense",
     {PHASE_ON}, ()),
    (DEFENSE_WEAK_DP, ".defense.weak_dp_defense", "WeakDPDefense",
     set(), ()),
    (DEFENSE_CCLIP, ".defense.cclip_defense", "CClipDefense",
     {PHASE_BEFORE, PHASE_AFTER}, ()),
    (DEFENSE_WISE_MEDIAN, ".defense.coordinate_wise_median_defense", "CoordinateWiseMedianDefense",
     {PHASE_ON}, ()),
    (DEFENSE_RFA, ".defense.RFA_defense", "RFADefense",
     {PHASE_ON}, ()),
    (DEFENSE_FOOLSGOLD, ".defense.foolsgold_defense", "FoolsGoldDefense",
     {PHASE_BEFORE}, ()),
    (DEFENSE_THREESIGMA_FOOLSGOLD, ".defense.three_sigma_defense_foolsgold", "ThreeSigmaDefense_Foolsgold",
     {PHASE_BEFORE}, ()),
    (DEFENSE_THREESIGMA_GEOMEDIAN, ".defense.three_sigma_geomedian_defense", "ThreeSigmaGeoMedianDefense",
     {PHASE_BEFORE}, ()),
    (DEFENSE_THREESIGMA, ".defense.three_sigma_defense", "ThreeSigmaDefense",
     {PHASE_BEFORE}, ()),
    (DEFENSE_CRFL, ".defense.crfl_defense", "CRFLDefense",
     {PHASE_AFTER}, ()),
    (DEFENSE_TRIMMED_MEAN, ".defense.coordinate_wise_trimmed_mean_defense", "CoordinateWiseTrimmedMeanDefense",
     {PHASE_BEFORE}, ()),
    (ANOMALY_DETECTION, ".defense.outlier_detection", "OutlierDetection",
     {PHASE_BEFORE}, ()),
]

for _dtype, _mod, _cls, _phases, _aliases in _BUILTIN:
    _DEFENSE_REGISTRY.register_lazy(_dtype, _mod, _cls, _phases, _aliases)


class FedMLDefender:
    _defender_instance = None

    @staticmethod
    def get_instance():
        if FedMLDefender._defender_instance is None:
            FedMLDefender._defender_instance = FedMLDefender()

        return FedMLDefender._defender_instance

    def __init__(self):
        self.is_enabled = False
        self.defense_type = None
        self.defender = None
        self._phases: frozenset = frozenset()

    # -- public API: register custom defenses --------------------------------

    @staticmethod
    def register_defense(defense_type: str, cls: Type[BaseDefenseMethod],
                         phases: Set[str], aliases: Tuple[str, ...] = ()):
        """Register a custom defense so it can be selected via config.

        Example::

            from fedml.core.security.fedml_defender import FedMLDefender, PHASE_BEFORE

            class MyCustomDefense(BaseDefenseMethod):
                ...

            FedMLDefender.register_defense("my_custom", MyCustomDefense, {PHASE_BEFORE})

        After registration, set ``defense_type: my_custom`` in the YAML config.
        """
        _DEFENSE_REGISTRY.register(defense_type, cls, phases, aliases)

    @staticmethod
    def available_defenses():
        """Return a sorted list of all registered defense type strings."""
        return _DEFENSE_REGISTRY.available()

    # -- lifecycle -----------------------------------------------------------

    def init(self, args):
        # Always reset state to guarantee isolation between experiments
        self.defender = None
        self.defense_type = None
        self._phases = frozenset()

        if hasattr(args, "enable_defense") and args.enable_defense:
            self.args = args
            self.is_enabled = True
            self.defense_type = args.defense_type.strip()
            logging.info("------init defense...%s", self.defense_type)

            entry = _DEFENSE_REGISTRY.get(self.defense_type)
            self.defender = entry.cls(args)
            self._phases = entry.phases

            logging.info(
                "Defense '%s' initialised (class=%s, phases=%s)",
                self.defense_type,
                type(self.defender).__name__,
                sorted(self._phases),
            )
        else:
            self.is_enabled = False

        if (
            self.is_enabled
            and hasattr(args, MLEngineBackend.ml_engine_args_flag)
            and args.ml_engine
            in [
                MLEngineBackend.ml_engine_backend_tf,
                MLEngineBackend.ml_engine_backend_jax,
                MLEngineBackend.ml_engine_backend_mxnet,
            ]
        ):
            logging.info(
                "FedMLDefender is not supported for the machine learning engine: %s. "
                "We will support more engines in the future iteration." % args.ml_engine
            )
            self.is_enabled = False

    def is_defense_enabled(self):
        return self.is_enabled

    def defend(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        return self.defender.run(
            raw_client_grad_list, base_aggregation_func, extra_auxiliary_info
        )

    def is_defense_on_aggregation(self):
        return self.is_defense_enabled() and PHASE_ON in self._phases

    def is_defense_before_aggregation(self):
        return self.is_defense_enabled() and PHASE_BEFORE in self._phases

    def is_defense_after_aggregation(self):
        return self.is_defense_enabled() and PHASE_AFTER in self._phases

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        if self.is_defense_before_aggregation():
            return self.defender.defend_before_aggregation(
                raw_client_grad_list, extra_auxiliary_info
            )
        return raw_client_grad_list

    def defend_on_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        if self.is_defense_on_aggregation():
            return self.defender.defend_on_aggregation(
                raw_client_grad_list, base_aggregation_func, extra_auxiliary_info
            )
        return base_aggregation_func(args=self.args, raw_grad_list=raw_client_grad_list)

    def defend_after_aggregation(self, global_model):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        if self.is_defense_after_aggregation():
            return self.defender.defend_after_aggregation(global_model)
        return global_model

    def get_malicious_client_idxs(self):
        return self.defender.get_malicious_client_idxs()

    def get_benign_client_idxs(self, client_idxs):
        return [i for i in client_idxs if i not in self.defender.get_malicious_client_idxs()]
