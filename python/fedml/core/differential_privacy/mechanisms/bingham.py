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
The Bingham mechanism in differential privacy, for estimating the first eigenvector of a covariance matrix.
"""
from numbers import Real

import numpy as np

from fedml.core.differential_privacy.mechanisms.base import DPMechanism
from fedml.core.differential_privacy.utils import copy_docstring


class Bingham(DPMechanism):
    r"""
    The Bingham mechanism in differential privacy.

    Used to estimate the first eigenvector (associated with the largest eigenvalue) of a covariance matrix.

    Paper link: http://eprints.whiterose.ac.uk/123206/7/simbingham8.pdf

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    sensitivity : float, default: 1
        The sensitivity of the mechanism.  Must be in [0, ∞).

    """
    def __init__(self, *, epsilon, sensitivity=1.0):
        super().__init__(epsilon=epsilon, delta=0)
        self.sensitivity = self._check_sensitivity(sensitivity)

        self._rng = np.random.default_rng()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

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

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Value to be randomised must be a numpy array, got {type(value)}")
        if value.ndim != 2:
            raise ValueError(f"Array must be 2-dimensional, got {value.ndim} dimensions")
        if value.shape[0] != value.shape[1]:
            raise ValueError(f"Array must be square, got {value.shape[0]} x {value.shape[1]}")
        if not np.allclose(value, value.T):
            raise ValueError("Array must be symmetric, supplied array is not.")

        return True

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : numpy array
            The data to be randomised.

        Returns
        -------
        numpy array
            The randomised eigenvector.

        """
        self._check_all(value)

        eigvals, eigvecs = np.linalg.eigh(value)
        dims = value.shape[0]

        if dims == 1:
            return np.ones((1, 1))
        if self.sensitivity / self.epsilon == 0:
            return eigvecs[:, eigvals.argmax()]

        value_translated = self.epsilon * (eigvals.max() * np.eye(dims) - value) / 4 / self.sensitivity
        translated_eigvals = np.linalg.eigvalsh(value_translated)

        left, right, mid = 1, dims, (1 + dims) / 2
        old_interval_size = (right - left) * 2

        while right - left < old_interval_size:
            old_interval_size = right - left

            mid = (right + left) / 2
            f_mid = np.array([1 / (mid + 2 * eig) for eig in translated_eigvals]).sum()

            if f_mid <= 1:
                right = mid

            if f_mid >= 1:
                left = mid

        b_const = mid
        omega = np.eye(dims) + 2 * value_translated / b_const
        omega_inv = np.linalg.inv(omega)
        norm_const = np.exp(-(dims - b_const) / 2) * ((dims / b_const) ** (dims / 2))

        while True:
            rnd_vec = self._rng.multivariate_normal(np.zeros(dims), omega_inv / 4, size=4).sum(axis=0)
            unit_vec = rnd_vec / np.linalg.norm(rnd_vec)
            prob = np.exp(-unit_vec.dot(value_translated).dot(unit_vec)) / norm_const\
                / ((unit_vec.dot(omega).dot(unit_vec)) ** (dims / 2))

            if self._rng.random() <= prob:
                return unit_vec
