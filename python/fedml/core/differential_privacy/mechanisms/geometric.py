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
The classic geometric mechanism for differential privacy, and its derivatives.
"""
from numbers import Integral

import numpy as np

from fedml.core.differential_privacy.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from fedml.core.differential_privacy.utils import copy_docstring


class Geometric(DPMechanism):
    r"""
    The classic geometric mechanism for differential privacy, as first proposed by Ghosh, Roughgarden and Sundararajan.
    Extended to allow for non-unity sensitivity.

    Paper link: https://arxiv.org/pdf/0811.2841.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    """
    def __init__(self, *, epsilon, sensitivity=1):
        super().__init__(epsilon=epsilon, delta=0.0)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = - self.epsilon / self.sensitivity if self.sensitivity > 0 else - float("inf")

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

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super()._check_epsilon_delta(epsilon, delta)

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        self._check_all(value)

        leading_factor = (1 - np.exp(self._scale)) / (1 + np.exp(self._scale))
        geom_series = np.exp(self._scale) / (1 - np.exp(self._scale))

        return 2 * leading_factor * (geom_series + 3 * (geom_series ** 2) + 2 * (geom_series ** 3))

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : int
            The value to be randomised.

        Returns
        -------
        int
            The randomised value.

        """
        self._check_all(value)

        # Need to account for overlap of 0-value between distributions of different sign
        unif_rv = self._rng.random() - 0.5
        unif_rv *= 1 + np.exp(self._scale)
        sgn = -1 if unif_rv < 0 else 1

        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(np.round(value + sgn * np.floor(np.log(sgn * unif_rv) / self._scale)))


class GeometricTruncated(Geometric, TruncationAndFoldingMixin):
    r"""
    The truncated geometric mechanism, where values that fall outside a pre-described range are mapped back to the
    closest point within the range.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : int
        The lower bound of the mechanism.

    upper : int
        The upper bound of the mechanism.

    """
    def __init__(self, *, epsilon, sensitivity=1, lower, upper):
        super().__init__(epsilon=epsilon, sensitivity=sensitivity)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @classmethod
    def _check_bounds(cls, lower, upper):
        if not isinstance(lower, Integral) and abs(lower) != float("inf"):
            raise TypeError(f"Lower bound must be integer-valued, got {lower}")
        if not isinstance(upper, Integral) and abs(upper) != float("inf"):
            raise TypeError(f"Upper bound must be integer-valued, got {upper}")

        return super()._check_bounds(lower, upper)

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.bias)
    def variance(self, value):
        raise NotImplementedError

    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return int(np.round(self._truncate(noisy_value)))


class GeometricFolded(Geometric, TruncationAndFoldingMixin):
    r"""
    The folded geometric mechanism, where values outside a pre-described range are folded back toward the domain around
    the closest point within the domain.
    Half-integer bounds are permitted.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : int or float
        The lower bound of the mechanism.  Must be integer or half-integer -valued.

    upper : int or float
        The upper bound of the mechanism.  Must be integer or half-integer -valued.

    """
    def __init__(self, *, epsilon, sensitivity=1, lower, upper):
        super().__init__(epsilon=epsilon, sensitivity=sensitivity)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @classmethod
    def _check_bounds(cls, lower, upper):
        if not np.isclose(2 * lower, np.round(2 * lower)) or not np.isclose(2 * upper, np.round(2 * upper)):
            raise ValueError("Bounds must be integer or half-integer floats")

        return super()._check_bounds(lower, upper)

    def _fold(self, value):
        return super()._fold(int(np.round(value)))

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.bias)
    def variance(self, value):
        raise NotImplementedError

    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Geometric.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return int(np.round(self._fold(noisy_value)))
