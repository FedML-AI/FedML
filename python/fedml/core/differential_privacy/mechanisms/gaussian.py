"""
The classic Gaussian mechanism in differential privacy, and its derivatives.
"""
import secrets
from math import erf
from numbers import Real, Integral
import numpy as np
import fedml.core.differential_privacy.common.utils as utils


class Gaussian:
    r"""The Gaussian mechanism in differential privacy.
    This code refers to IBM DP Library: https://github.com/IBM/differential-privacy-library
    Our contribution: code refactoring; remove some redundant codes

    "The algorithmic foundations of differential privacy" [DR14]_."
    Samples from the Gaussian distribution are generated using two samples from `random.normalvariate` as in [HB21b]_,
    to prevent against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float. Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, 1].  For ``epsilon > 1``, use
        :class:`.GaussianAnalytic`.

    delta : float. Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : float. The sensitivity of the mechanism.  Must be in [0, ∞).

    References
    ----------
    .. [DR14] Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy."
    .. [HB21b] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy."
    """

    def __init__(self, *, epsilon, delta, sensitivity):
        utils.check_params(epsilon, delta, sensitivity)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        # special requirements for epsilon and delta in Gaussian
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")
        self.special_check_for_params()
        self._scale = (
            np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        )
        self._rng = secrets.SystemRandom()

    def special_check_for_params(self):
        if self.epsilon > 1.0:
            raise ValueError(
                "Epsilon cannot be greater than 1. If required, use GaussianAnalytic instead."
            )

    def bias(self, value):
        return 0.0

    def variance(self, value):
        return self._scale**2

    def randomise(self, value):
        utils.check_numeric_value(value)
        standard_normal = (
            self._rng.normalvariate(0, 1) + self._rng.normalvariate(0, 1)
        ) / np.sqrt(2)
        return value + standard_normal * self._scale


class GaussianAnalytic(Gaussian):
    r"""The analytic Gaussian mechanism in differential privacy.
    "Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising".
     https://arxiv.org/pdf/1805.06530.pdf
    """

    def __init__(self, *, epsilon, delta, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        self._scale = self._find_scale()

    def special_check_for_params(self):
        pass

    def _find_scale(self):
        if self.sensitivity / self.epsilon == 0:
            return 0.0

        epsilon = self.epsilon
        delta = self.delta

        def phi(val):
            return (1 + erf(val / np.sqrt(2))) / 2

        def b_plus(val):
            return (
                phi(np.sqrt(epsilon * val))
                - np.exp(epsilon) * phi(-np.sqrt(epsilon * (val + 2)))
                - delta
            )

        def b_minus(val):
            return (
                phi(-np.sqrt(epsilon * val))
                - np.exp(epsilon) * phi(-np.sqrt(epsilon * (val + 2)))
                - delta
            )

        delta_0 = b_plus(0)

        if delta_0 == 0:
            alpha = 1
        else:
            if delta_0 < 0:
                target_func = b_plus
            else:
                target_func = b_minus

            # Find the starting interval by doubling the initial size until the target_func sign changes, as suggested
            # in the paper
            left = 0
            right = 1
            while target_func(left) * target_func(right) > 0:
                left = right
                right *= 2

            # Binary search code copied from mechanisms.LaplaceBoundedDomain
            old_interval_size = (right - left) * 2
            while old_interval_size > right - left:
                old_interval_size = right - left
                middle = (right + left) / 2
                if target_func(middle) * target_func(left) <= 0:
                    right = middle
                if target_func(middle) * target_func(right) <= 0:
                    left = middle
            alpha = np.sqrt(1 + (left + right) / 4) + (
                -1 if delta_0 < 0 else 1
            ) * np.sqrt((left + right) / 4)
        return alpha * self.sensitivity / np.sqrt(2 * self.epsilon)


class GaussianDiscrete(Gaussian):
    r"""The Discrete Gaussian mechanism in differential privacy.
    The Discrete Gaussian for Differential Privacy: https://arxiv.org/pdf/2004.00010.pdf

    Parameters
    ----------
    epsilon : float. Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].
    delta : float. Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].
    sensitivity : int, default: 1. The sensitivity of the mechanism.  Must be in [0, ∞).
    """

    def __init__(self, *, epsilon, delta, sensitivity=1):
        if not isinstance(sensitivity, Integral):
            raise TypeError("Sensitivity must be an integer")
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        self._scale = self._find_scale()

    def _check_epsilon_delta(self):
        pass

    def bias(self, value):
        return 0.0

    def randomise(self, value):
        utils.check_integer_value(value)
        if self._scale == 0:
            return value
        tau = 1 / (1 + np.floor(self._scale))
        sigma2 = self._scale**2
        while True:
            geom_x = 0
            while utils.bernoulli_neg_exp(tau, self._rng):
                geom_x += 1
            bern_b = np.random.binomial(1, 0.5)
            if bern_b and not geom_x:
                continue
            lap_y = int((1 - 2 * bern_b) * geom_x)
            bern_c = utils.bernoulli_neg_exp(
                (abs(lap_y) - tau * sigma2) ** 2 / 2 / sigma2, self._rng
            )
            if bern_c:
                return value + lap_y

    def _find_scale(self):
        """Determine the scale of the mechanism's distribution given epsilon and delta."""
        if self.sensitivity / self.epsilon == 0:
            return 0

        def objective(sigma, epsilon_, delta_, sensitivity_):
            """Function for which we are seeking its root."""
            idx_0 = int(
                np.floor(epsilon_ * sigma**2 / sensitivity_ - sensitivity_ / 2)
            )
            idx_1 = int(
                np.floor(epsilon_ * sigma**2 / sensitivity_ + sensitivity_ / 2)
            )
            idx = 1

            lhs, rhs, denom = float(idx_0 < 0), 0, 1
            _term, diff = 1, 1

            while _term > 0 and diff > 0:
                _term = np.exp(-(idx**2) / 2 / sigma**2)

                if idx > idx_0:
                    lhs += _term

                    if idx_0 < -idx:
                        lhs += _term

                    if idx > idx_1:
                        diff = -rhs
                        rhs += _term
                        diff += rhs

                denom += 2 * _term
                idx += 1
                if idx > 1e6:
                    raise ValueError(
                        "Infinite sum not converging, aborting. Try changing the epsilon and/or delta."
                    )
            return (lhs - np.exp(epsilon_) * rhs) / denom - delta_

        epsilon = self.epsilon
        delta = self.delta
        sensitivity = self.sensitivity

        # Begin by locating the root within an interval [2**i, 2**(i+1)]
        guess_0 = 1
        f_0 = objective(guess_0, epsilon, delta, sensitivity)
        pwr = 1 if f_0 > 0 else -1
        guess_1 = 2**pwr
        f_1 = objective(guess_1, epsilon, delta, sensitivity)

        while f_0 * f_1 > 0:
            guess_0 *= 2**pwr
            guess_1 *= 2**pwr
            f_0 = f_1
            f_1 = objective(guess_1, epsilon, delta, sensitivity)

        # Find the root (sigma) using the bisection method
        while not np.isclose(guess_0, guess_1, atol=1e-12, rtol=1e-6):
            guess_mid = (guess_0 + guess_1) / 2
            f_mid = objective(guess_mid, epsilon, delta, sensitivity)
            if f_mid * f_0 <= 0:
                f_1 = f_mid
                guess_1 = guess_mid
            if f_mid * f_1 <= 0:
                f_0 = f_mid
                guess_0 = guess_mid

        return (guess_0 + guess_1) / 2
