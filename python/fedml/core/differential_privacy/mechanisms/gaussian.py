# MIT License
#
# Copyright (C) IBM Corporation 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The classic Gaussian mechanism in differential privacy, and its derivatives.
"""
from math import erf
from numbers import Real, Integral

import numpy as np

from fedml.core.differential_privacy.mechanisms.base import DPMechanism, bernoulli_neg_exp
from fedml.core.differential_privacy.mechanisms.geometric import Geometric
from fedml.core.differential_privacy.mechanisms.laplace import Laplace
from fedml.core.differential_privacy.utils import copy_docstring


class Gaussian(DPMechanism):
    r"""The Gaussian mechanism in differential privacy.

    First proposed by Dwork and Roth in "The algorithmic foundations of differential privacy" [DR14]_.  Samples from the
    Gaussian distribution are generated using two samples from `random.normalvariate` as detailed in [HB21b]_, to
    prevent against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, 1].  For ``epsilon > 1``, use
        :class:`.GaussianAnalytic`.

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    References
    ----------
    .. [DR14] Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy." Found. Trends
        Theor. Comput. Sci. 9, no. 3-4 (2014): 211-407.

    .. [HB21b] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy." arXiv preprint
        arXiv:2107.10138 (2021).

    """
    def __init__(self, *, epsilon, delta, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        if isinstance(epsilon, Real) and epsilon > 1.0:
            raise ValueError("Epsilon cannot be greater than 1. If required, use GaussianAnalytic instead.")

        return super()._check_epsilon_delta(epsilon, delta)

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return float(sensitivity)

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        return True

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(0)

        return self._scale ** 2

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        standard_normal = (self._rng.normalvariate(0, 1) + self._rng.normalvariate(0, 1)) / np.sqrt(2)

        return value + standard_normal * self._scale


class GaussianAnalytic(Gaussian):
    r"""The analytic Gaussian mechanism in differential privacy.

    As first proposed by Balle and Wang in "Improving the Gaussian Mechanism for Differential Privacy: Analytical
    Calibration and Optimal Denoising".

    Paper link: https://arxiv.org/pdf/1805.06530.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    """
    def __init__(self, *, epsilon, delta, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        self._scale = self._find_scale()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        return super(Gaussian, cls)._check_epsilon_delta(epsilon, delta)

    def _check_all(self, value):
        super()._check_all(value)

        return True

    def _find_scale(self):
        if self.sensitivity / self.epsilon == 0:
            return 0.0

        epsilon = self.epsilon
        delta = self.delta

        def phi(val):
            return (1 + erf(val / np.sqrt(2))) / 2

        def b_plus(val):
            return phi(np.sqrt(epsilon * val)) - np.exp(epsilon) * phi(- np.sqrt(epsilon * (val + 2))) - delta

        def b_minus(val):
            return phi(- np.sqrt(epsilon * val)) - np.exp(epsilon) * phi(- np.sqrt(epsilon * (val + 2))) - delta

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

            alpha = np.sqrt(1 + (left + right) / 4) + (-1 if delta_0 < 0 else 1) * np.sqrt((left + right) / 4)

        return alpha * self.sensitivity / np.sqrt(2 * self.epsilon)


class GaussianDiscrete(DPMechanism):
    r"""The Discrete Gaussian mechanism in differential privacy.

    As proposed by Canonne, Kamath and Steinke, re-purposed for approximate :math:`(\epsilon,\delta)`-differential
    privacy.

    Paper link: https://arxiv.org/pdf/2004.00010.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 1].

    sensitivity : int, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    """
    def __init__(self, *, epsilon, delta, sensitivity=1):
        super().__init__(epsilon=epsilon, delta=delta)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = self._find_scale()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0 or delta == 0:
            raise ValueError("Neither Epsilon nor Delta can be zero")

        return super()._check_epsilon_delta(epsilon, delta)

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Integral):
            raise TypeError("Sensitivity must be an integer")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return sensitivity

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        if not isinstance(value, Integral):
            raise TypeError("Value to be randomised must be an integer")

        return True

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        raise NotImplementedError

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        self._check_all(value)

        if self._scale == 0:
            return value

        tau = 1 / (1 + np.floor(self._scale))
        sigma2 = self._scale ** 2

        while True:
            geom_x = 0
            while bernoulli_neg_exp(tau, self._rng):
                geom_x += 1

            bern_b = np.random.binomial(1, 0.5)
            if bern_b and not geom_x:
                continue

            lap_y = int((1 - 2 * bern_b) * geom_x)
            bern_c = bernoulli_neg_exp((abs(lap_y) - tau * sigma2) ** 2 / 2 / sigma2, self._rng)
            if bern_c:
                return value + lap_y

    def _find_scale(self):
        """Determine the scale of the mechanism's distribution given epsilon and delta.
        """
        if self.sensitivity / self.epsilon == 0:
            return 0

        def objective(sigma, epsilon_, delta_, sensitivity_):
            """Function for which we are seeking its root. """
            idx_0 = int(np.floor(epsilon_ * sigma ** 2 / sensitivity_ - sensitivity_ / 2))
            idx_1 = int(np.floor(epsilon_ * sigma ** 2 / sensitivity_ + sensitivity_ / 2))
            idx = 1

            lhs, rhs, denom = float(idx_0 < 0), 0, 1
            _term, diff = 1, 1

            while _term > 0 and diff > 0:
                _term = np.exp(-idx ** 2 / 2 / sigma ** 2)

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
                    raise ValueError("Infinite sum not converging, aborting. Try changing the epsilon and/or delta.")

            return (lhs - np.exp(epsilon_) * rhs) / denom - delta_

        epsilon = self.epsilon
        delta = self.delta
        sensitivity = self.sensitivity

        # Begin by locating the root within an interval [2**i, 2**(i+1)]
        guess_0 = 1
        f_0 = objective(guess_0, epsilon, delta, sensitivity)
        pwr = 1 if f_0 > 0 else -1
        guess_1 = 2 ** pwr
        f_1 = objective(guess_1, epsilon, delta, sensitivity)

        while f_0 * f_1 > 0:
            guess_0 *= 2 ** pwr
            guess_1 *= 2 ** pwr

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
