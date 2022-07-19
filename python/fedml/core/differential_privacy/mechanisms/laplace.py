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
The classic Laplace mechanism in differential privacy, and its derivatives.
"""
from numbers import Real

import numpy as np

from fedml.core.differential_privacy.mechanisms.base import DPMechanism, TruncationAndFoldingMixin
from fedml.core.differential_privacy.utils import copy_docstring


class Laplace(DPMechanism):
    r"""
    The classical Laplace mechanism in differential privacy.

    First proposed by Dwork, McSherry, Nissim and Smith [DMNS16]_, with support for (relaxed)
    :math:`(\epsilon,\delta)`-differential privacy [HLM15]_.

    Samples from the Laplace distribution are generated using 4 uniform variates, as detailed in [HB21]_, to prevent
    against reconstruction attacks due to limited floating point precision.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    References
    ----------
    .. [DMNS16] Dwork, Cynthia, Frank McSherry, Kobbi Nissim, and Adam Smith. "Calibrating noise to sensitivity in
        private data analysis." Journal of Privacy and Confidentiality 7, no. 3 (2016): 17-51.

    .. [HLM15] Holohan, Naoise, Douglas J. Leith, and Oliver Mason. "Differential privacy in metric spaces: Numerical,
        categorical and functional data under the one roof." Information Sciences 305 (2015): 256-268.

    .. [HB21] Holohan, Naoise, and Stefano Braghin. "Secure Random Sampling in Differential Privacy." arXiv preprint
        arXiv:2107.10138 (2021).

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = None

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

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`.

        Parameters
        ----------
        value : int or float
            The value at which the bias of the mechanism is sought.

        Returns
        -------
        bias : float or None
            The bias of the mechanism at `value`.

        """
        return 0.0

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`.

        Parameters
        ----------
        value : float
            The value at which the variance of the mechanism is sought.

        Returns
        -------
        bias : float
            The variance of the mechanism at `value`.

        """
        self._check_all(0)

        return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(np.pi * unif4)

    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : float
            The value to be randomised.

        Returns
        -------
        float
            The randomised value.

        """
        self._check_all(value)

        scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
        standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
                                                 self._rng.random())

        return value - scale * standard_laplace


class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
    r"""
    The truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point
    within the domain.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        return shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape))

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        variance = value ** 2 + shape * (self.lower * np.exp((self.lower - value) / shape)
                                         - self.upper * np.exp((value - self.upper) / shape))
        variance += (shape ** 2) * (2 - np.exp((self.lower - value) / shape)
                                    - np.exp((value - self.upper) / shape))

        variance -= (self.bias(value) + value) ** 2

        return variance

    def _check_all(self, value):
        Laplace._check_all(self, value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return self._truncate(noisy_value)


class LaplaceFolded(Laplace, TruncationAndFoldingMixin):
    r"""
    The folded Laplace mechanism, where values outside a pre-described domain are folded around the domain until they
    fall within.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        bias = shape * (np.exp((self.lower + self.upper - 2 * value) / shape) - 1)
        bias /= np.exp((self.lower - value) / shape) + np.exp((self.upper - value) / shape)

        return bias

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def _check_all(self, value):
        super()._check_all(value)
        TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return self._fold(noisy_value)


class LaplaceBoundedDomain(LaplaceTruncated):
    r"""
    The bounded Laplace mechanism on a bounded domain.  The mechanism draws values directly from the domain using
    rejection sampling, without any post-processing [HABM20]_.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in [0, ∞].

    delta : float, default: 0.0
        Privacy parameter :math:`\delta` for the mechanism.  Must be in [0, 1].  Cannot be simultaneously zero with
        ``epsilon``.

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    lower : float
        The lower bound of the mechanism.

    upper : float
        The upper bound of the mechanism.

    References
    ----------
    .. [HABM20] Holohan, Naoise, Spiros Antonatos, Stefano Braghin, and Pól Mac Aonghusa. "The Bounded Laplace Mechanism
        in Differential Privacy." Journal of Privacy and Confidentiality 10, no. 1 (2020).

    """
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, lower=lower, upper=upper)
        self._rng = np.random.default_rng()

    def _find_scale(self):
        eps = self.epsilon
        delta = self.delta
        diam = self.upper - self.lower
        delta_q = self.sensitivity

        def _delta_c(shape):
            if shape == 0:
                return 2.0
            return (2 - np.exp(- delta_q / shape) - np.exp(- (diam - delta_q) / shape)) / (1 - np.exp(- diam / shape))

        def _f(shape):
            return delta_q / (eps - np.log(_delta_c(shape)) - np.log(1 - delta))

        left = delta_q / (eps - np.log(1 - delta))
        right = _f(left)
        old_interval_size = (right - left) * 2

        while old_interval_size > right - left:
            old_interval_size = right - left
            middle = (right + left) / 2

            if _f(middle) >= middle:
                left = middle
            if _f(middle) <= middle:
                right = middle

        return (right + left) / 2

    def effective_epsilon(self):
        r"""Gets the effective epsilon of the mechanism, only for strict :math:`\epsilon`-differential privacy.  Returns
        ``None`` if :math:`\delta` is non-zero.

        Returns
        -------
        float
            The effective :math:`\epsilon` parameter of the mechanism.  Returns ``None`` if `delta` is non-zero.

        """
        if self._scale is None:
            self._scale = self._find_scale()

        if self.delta > 0.0:
            return None

        return self.sensitivity / self._scale

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        bias = (self._scale - self.lower + value) / 2 * np.exp((self.lower - value) / self._scale) \
            - (self._scale + self.upper - value) / 2 * np.exp((value - self.upper) / self._scale)
        bias /= 1 - np.exp((self.lower - value) / self._scale) / 2 \
            - np.exp((value - self.upper) / self._scale) / 2

        return bias

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        variance = value**2
        variance -= (np.exp((self.lower - value) / self._scale) * (self.lower ** 2)
                     + np.exp((value - self.upper) / self._scale) * (self.upper ** 2)) / 2
        variance += self._scale * (self.lower * np.exp((self.lower - value) / self._scale)
                                   - self.upper * np.exp((value - self.upper) / self._scale))
        variance += (self._scale ** 2) * (2 - np.exp((self.lower - value) / self._scale)
                                          - np.exp((value - self.upper) / self._scale))
        variance /= 1 - (np.exp(-(value - self.lower) / self._scale)
                         + np.exp(-(self.upper - value) / self._scale)) / 2

        variance -= (self.bias(value) + value) ** 2

        return variance

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        if self._scale is None:
            self._scale = self._find_scale()

        value = max(min(value, self.upper), self.lower)
        if np.isnan(value):
            return float("nan")

        samples = 1

        while True:
            noisy = value + self._scale * self._laplace_sampler(self._rng.random(samples), self._rng.random(samples),
                                                                self._rng.random(samples), self._rng.random(samples))
            if ((noisy >= self.lower) & (noisy <= self.upper)).any():
                idx = np.argmax((noisy >= self.lower) & (noisy <= self.upper))
                return noisy[idx]
            samples = min(100000, samples * 2)


class LaplaceBoundedNoise(Laplace):
    r"""
    The Laplace mechanism with bounded noise, only applicable for approximate differential privacy (delta > 0)
    [GDGK18]_.

    Epsilon must be strictly positive, `epsilon` > 0. `delta` must be strictly in the interval (0, 0.5).
     - For zero `epsilon`, use :class:`.Uniform`.
     - For zero `delta`, use :class:`.Laplace`.

    Parameters
    ----------
    epsilon : float
        Privacy parameter :math:`\epsilon` for the mechanism.  Must be in (0, ∞].

    delta : float
        Privacy parameter :math:`\delta` for the mechanism.  Must be in (0, 0.5).

    sensitivity : float
        The sensitivity of the mechanism.  Must be in [0, ∞).

    References
    ----------
    .. [GDGK18] Geng, Quan, Wei Ding, Ruiqi Guo, and Sanjiv Kumar. "Truncated Laplacian Mechanism for Approximate
        Differential Privacy." arXiv preprint arXiv:1810.00877v1 (2018).

    """
    def __init__(self, *, epsilon, delta, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        self._noise_bound = None
        self._rng = np.random.default_rng()

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if epsilon == 0:
            raise ValueError("Epsilon must be strictly positive. For zero epsilon, use :class:`.Uniform`.")

        if isinstance(delta, Real) and not 0 < delta < 0.5:
            raise ValueError("Delta must be strictly in the interval (0,0.5). For zero delta, use :class:`.Laplace`.")

        return super(Laplace, cls)._check_epsilon_delta(epsilon, delta)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        return 0.0

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        if self._scale is None or self._noise_bound is None:
            self._scale = self.sensitivity / self.epsilon
            self._noise_bound = 0 if self._scale == 0 else \
                self._scale * np.log(1 + (np.exp(self.epsilon) - 1) / 2 / self.delta)

        if np.isnan(value):
            return float("nan")

        samples = 1

        while True:
            noisy = self._scale * self._laplace_sampler(self._rng.random(samples), self._rng.random(samples),
                                                        self._rng.random(samples), self._rng.random(samples))
            if ((noisy >= - self._noise_bound) & (noisy <= self._noise_bound)).any():
                idx = np.argmax((noisy >= - self._noise_bound) & (noisy <= self._noise_bound))
                return value + noisy[idx]
            samples = min(100000, samples * 2)
