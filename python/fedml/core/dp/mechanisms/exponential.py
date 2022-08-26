"""
Implementation of the standard exponential mechanism, and its derivative, the hierarchical mechanism.
"""
import secrets
from numbers import Real
import numpy as np
from fedml.core.dp.common.utils import bernoulli_neg_exp, check_params


class Exponential:
    r"""
    The exponential mechanism for achieving differential privacy on candidate selection, as first proposed by McSherry
    and Talwar. This code refers to IBM DP Library: https://github.com/IBM/differential-privacy-library

    The exponential mechanism achieves differential privacy by randomly choosing a candidate subject to candidate
    utility scores, with greater probability given to higher-utility candidates.

    Paper link: https://www.cs.drexel.edu/~greenie/privacy/mdviadp.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float
        The sensitivity in utility values to a change in a datapoint in the underlying dataset.

    utility : list
        A list of non-negative utility values for each candidate.

    monotonic : bool, default: False
        Specifies if the utility function is monotonic, i.e. that adding an individual to the underlying dataset can
        only increase the values in `utility`.

    candidates : list, optional
        An optional list of candidate labels.  If omitted, the zero-indexed list [0, 1, ..., n] is used.

    measure : list, optional
        An optional list of measures for each candidate.  If omitted, a uniform measure is used.

    """

    def __init__(
        self,
        *,
        epsilon,
        sensitivity,
        utility,
        monotonic=False,
        candidates=None,
        measure=None,
    ):
        check_params(epsilon, delta=0.00, sensitivity=sensitivity)
        self.epsilon = epsilon
        self.sensitivity = float(sensitivity)
        self.utility, self.candidates, self.measure = self.special_check_for_params(
            utility, candidates, measure
        )
        self.monotonic = bool(monotonic)
        self._probabilities = self._find_probabilities(
            self.epsilon, self.sensitivity, self.utility, self.monotonic, self.measure
        )
        self._rng = secrets.SystemRandom()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise TypeError("Epsilon and delta must be numeric")

        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0.0 <= float(delta) <= 1.0:
            raise ValueError("Delta must be in [0, 1]")

        if epsilon + delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        return float(epsilon), float(delta)

    @classmethod
    def special_check_for_params(cls, utility, candidates, measure):
        if not isinstance(utility, list):
            raise TypeError(f"Utility must be a list, got a {utility}.")

        if not all(isinstance(u, Real) for u in utility):
            raise TypeError("Utility must be a list of real-valued numbers.")

        if len(utility) < 1:
            raise ValueError("Utility must have at least one element.")

        if np.isinf(utility).any():
            raise ValueError("Utility must be a list of finite numbers.")

        if candidates is not None:
            if not isinstance(candidates, list):
                raise TypeError(f"Candidates must be a list, got a {type(candidates)}.")

            if len(candidates) != len(utility):
                raise ValueError(
                    "List of candidates must be the same length as the list of utility values."
                )

        if measure is not None:
            if not isinstance(measure, list):
                raise TypeError(f"Measure must be a list, got a {type(measure)}.")

            if not all(isinstance(m, Real) for m in measure):
                raise TypeError("Measure must be a list of real-valued numbers.")

            if np.isinf(measure).any():
                raise ValueError("Measure must be a list of finite numbers.")

            if len(measure) != len(utility):
                raise ValueError(
                    "List of measures must be the same length as the list of utility values."
                )

        return utility, candidates, measure

    @classmethod
    def _find_probabilities(cls, epsilon, sensitivity, utility, monotonic, measure):
        scale = (
            epsilon / sensitivity / (2 - monotonic)
            if sensitivity / epsilon > 0
            else float("inf")
        )

        # Set max utility to 0 to avoid overflow on high utility; will be normalised out before returning
        utility = np.array(utility) - max(utility)

        if np.isinf(scale):
            probabilities = np.isclose(utility, 0).astype(float)
        else:
            probabilities = np.exp(scale * utility)

        probabilities *= np.array(measure) if measure else 1
        probabilities /= probabilities.sum()

        return np.cumsum(probabilities)

    def _check_all(self, value):
        if not isinstance(self.epsilon, Real) :
            raise TypeError("Epsilon and delta must be numeric")
        if self.epsilon < 0:
            raise ValueError("Epsilon must be non-negative")
        self._check_sensitivity(self.sensitivity)
        self._check_utility_candidates_measure(
            self.utility, self.candidates, self.measure
        )
        if value is not None:
            raise ValueError(f"Value to be randomised must be None. Got: {value}.")

        return True

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise TypeError("Epsilon and delta must be numeric")

        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= float(delta) <= 1:
            raise ValueError("Delta must be in [0, 1]")

        if epsilon + delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        return float(epsilon), float(delta)

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return float(sensitivity)

    @classmethod
    def _check_utility_candidates_measure(cls, utility, candidates, measure):
        if not isinstance(utility, list):
            raise TypeError(f"Utility must be a list, got a {utility}.")

        if not all(isinstance(u, Real) for u in utility):
            raise TypeError("Utility must be a list of real-valued numbers.")

        if len(utility) < 1:
            raise ValueError("Utility must have at least one element.")

        if np.isinf(utility).any():
            raise ValueError("Utility must be a list of finite numbers.")

        if candidates is not None:
            if not isinstance(candidates, list):
                raise TypeError(f"Candidates must be a list, got a {type(candidates)}.")

            if len(candidates) != len(utility):
                raise ValueError("List of candidates must be the same length as the list of utility values.")

        if measure is not None:
            if not isinstance(measure, list):
                raise TypeError(f"Measure must be a list, got a {type(measure)}.")

            if not all(isinstance(m, Real) for m in measure):
                raise TypeError("Measure must be a list of real-valued numbers.")

            if np.isinf(measure).any():
                raise ValueError("Measure must be a list of finite numbers.")

            if len(measure) != len(utility):
                raise ValueError("List of measures must be the same length as the list of utility values.")

        return utility, candidates, measure

    def bias(self, value):
        raise NotImplementedError

    def variance(self, value):
        raise NotImplementedError

    def randomise(self, value=None):
        """Select a candidate with differential privacy.

        Parameters
        ----------
        value : None
            Ignored.

        Returns
        -------
        int or other
            The randomised candidate.

        """
        self._check_all(value)

        rand = self._rng.random()

        if np.any(rand <= self._probabilities):
            idx = np.argmax(rand <= self._probabilities)
        elif np.isclose(rand, self._probabilities[-1]):
            idx = len(self._probabilities) - 1
        else:
            raise RuntimeError(
                "Can't find a candidate to return. "
                f"Debugging info: Rand: {rand}, Probabilities: {self._probabilities}"
            )

        return self.candidates[idx] if self.candidates else idx
