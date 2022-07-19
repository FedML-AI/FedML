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
The uniform mechanism in differential privacy.
"""
from numbers import Real

from fedml.core.differential_privacy.mechanisms.base import DPMechanism
from fedml.core.differential_privacy.mechanisms.laplace import Laplace
from fedml.core.differential_privacy.utils import copy_docstring


class Uniform(DPMechanism):
    r"""
    The Uniform mechanism in differential privacy.

    This emerges as a special case of the :class:`.LaplaceBoundedNoise` mechanism when epsilon = 0.
    Paper link: https://arxiv.org/pdf/1810.00877.pdf

    Parameters
    ----------
    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 0.5].

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    """
    def __init__(self, *, delta, sensitivity):
        super().__init__(epsilon=0.0, delta=delta)
        self.sensitivity = self._check_sensitivity(sensitivity)

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not epsilon == 0:
            raise ValueError("Epsilon must be strictly zero.")

        if not 0 < delta <= 0.5:
            raise ValueError("Delta must be in the half-open interval (0, 0.5]")

        return super()._check_epsilon_delta(epsilon, delta)

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity < 0:
            raise ValueError("Sensitivity must be non-negative")

        return float(sensitivity)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        return (self.sensitivity / self.delta) ** 2 / 12

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        if not isinstance(value, Real):
            raise TypeError("Value to be randomised must be a number")

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        unif_rv = 2 * self._rng.random() - 1
        unif_rv *= self.sensitivity / self.delta / 2

        return value + unif_rv
