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
The vector mechanism in differential privacy, for producing perturbed objectives
"""
from numbers import Real

import numpy as np

from fedml.core.differential_privacy.mechanisms.base import DPMechanism
from fedml.core.differential_privacy.utils import copy_docstring


class Vector(DPMechanism):
    r"""
    The vector mechanism in differential privacy.

    The vector mechanism is used when perturbing convex objective functions.
    Full paper: http://www.jmlr.org/papers/volume12/chaudhuri11a/chaudhuri11a.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    function_sensitivity : float
        The function sensitivity of the mechanism.  Must be in [0, ∞).

    data_sensitivity : float, default: 1.0
        The data sensitivityof the mechanism.  Must be in [0, ∞).

    dimension : int
        Function input dimension.  This dimension relates to the size of the input vector of the function being
        considered by the mechanism.  This corresponds to the size of the random vector produced by the mechanism. Must
        be in [1, ∞).

    alpha : float, default: 0.01
        Regularisation parameter.  Must be in (0, ∞).

    """
    def __init__(self, *, epsilon, function_sensitivity, data_sensitivity=1.0, dimension, alpha=0.01):
        super().__init__(epsilon=epsilon, delta=0.0)
        self.function_sensitivity, self.data_sensitivity = self._check_sensitivity(function_sensitivity,
                                                                                   data_sensitivity)
        self.dimension = self._check_dimension(dimension)
        self.alpha = self._check_alpha(alpha)

        self._rng = np.random.default_rng()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super()._check_epsilon_delta(epsilon, delta)

    @classmethod
    def _check_alpha(cls, alpha):
        if not isinstance(alpha, Real):
            raise TypeError("Alpha must be numeric")

        if alpha <= 0:
            raise ValueError("Alpha must be strictly positive")

        return alpha

    @classmethod
    def _check_dimension(cls, vector_dim):
        if not isinstance(vector_dim, Real) or not np.isclose(vector_dim, int(vector_dim)):
            raise TypeError("d must be integer-valued")
        if int(vector_dim) < 1:
            raise ValueError("d must be strictly positive")

        return int(vector_dim)

    @classmethod
    def _check_sensitivity(cls, function_sensitivity, data_sensitivity):
        if not isinstance(function_sensitivity, Real) or not isinstance(data_sensitivity, Real):
            raise TypeError("Sensitivities must be numeric")

        if function_sensitivity < 0 or data_sensitivity < 0:
            raise ValueError("Sensitivities must be non-negative")

        return function_sensitivity, data_sensitivity

    def _check_all(self, value):
        super()._check_all(value)
        self._check_alpha(self.alpha)
        self._check_sensitivity(self.function_sensitivity, self.data_sensitivity)
        self._check_dimension(self.dimension)

        if not callable(value):
            raise TypeError("Value to be randomised must be a function")

        return True

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        If `value` is a method of two outputs, they are taken as `f` and `fprime` (i.e., its gradient), and both are
        perturbed accordingly.

        Parameters
        ----------
        value : method
            The function to be randomised.

        Returns
        -------
        method
            The randomised method.

        """
        self._check_all(value)

        epsilon_p = self.epsilon - 2 * np.log(1 + self.function_sensitivity * self.data_sensitivity /
                                              (0.5 * self.alpha))
        delta = 0

        if epsilon_p <= 0:
            delta = (self.function_sensitivity * self.data_sensitivity / (np.exp(self.epsilon / 4) - 1)
                     - 0.5 * self.alpha)
            epsilon_p = self.epsilon / 2

        scale = self.data_sensitivity * 2 / epsilon_p

        normed_noisy_vector = self._rng.standard_normal((self.dimension, 4)).sum(axis=1) / 2
        norm = np.linalg.norm(normed_noisy_vector, 2)
        noisy_norm = self._rng.gamma(self.dimension / 4, scale, 4).sum()

        normed_noisy_vector = normed_noisy_vector / norm * noisy_norm

        def output_func(*args):
            input_vec = args[0]

            func = value(*args)

            if isinstance(func, tuple):
                func, grad = func
            else:
                grad = None

            func += np.dot(normed_noisy_vector, input_vec)
            func += 0.5 * delta * np.dot(input_vec, input_vec)

            if grad is not None:
                grad += normed_noisy_vector + delta * input_vec

                return func, grad

            return func

        return output_func
