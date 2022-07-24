import secrets
import numpy as np
import fedml.core.differential_privacy.common.utils as utils


class Laplace:
    """
    The classical Laplace mechanism in differential privacy. This code refers to IBM DP Library: https://github.com/IBM/differential-privacy-library
    Our contribution: code refactoring; remove some redundant codes

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
        utils.check_params(epsilon, delta, sensitivity)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self._rng = secrets.SystemRandom()

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`."""
        return 0.0

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`."""
        return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(
            np.pi * unif4
        )

    def randomise(self, value):
        """Randomise `value` with the mechanism."""
        utils.check_numeric_value(value)
        scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
        standard_laplace = self._laplace_sampler(
            self._rng.random(),
            self._rng.random(),
            self._rng.random(),
            self._rng.random(),
        )
        return value - scale * standard_laplace


class LaplaceTruncated(Laplace):
    """
    The truncated Laplace mechanism, where values outside a pre-described domain are mapped to the closest point
    within the domain.
    """

    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower_bound, upper_bound):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        self.lower_bound, self.upper_bound = utils.check_bounds(
            lower_bound, upper_bound
        )

    def bias(self, value):
        utils.check_numeric_value(value)
        shape = self.sensitivity / self.epsilon
        return (
            shape
            / 2
            * (
                np.exp((self.lower_bound - value) / shape)
                - np.exp((value - self.upper_bound) / shape)
            )
        )

    def variance(self, value):
        utils.check_numeric_value(value)
        shape = self.sensitivity / self.epsilon
        variance = value**2 + shape * (
            self.lower_bound * np.exp((self.lower_bound - value) / shape)
            - self.upper_bound * np.exp((value - self.upper_bound) / shape)
        )
        variance += (shape**2) * (
            2
            - np.exp((self.lower_bound - value) / shape)
            - np.exp((value - self.upper_bound) / shape)
        )
        variance -= (self.bias(value) + value) ** 2
        return variance

    def randomise(self, value):
        utils.check_numeric_value(value)
        noisy_value = super().randomise(value)
        return self._truncate(noisy_value)

    def _truncate(self, value):
        if value > self.upper_bound:
            return self.upper_bound
        if value < self.lower_bound:
            return self.lower_bound
        return value


class LaplaceFolded(Laplace):
    """
    The folded Laplace mechanism, where values outside a pre-described domain are folded around the domain until they
    fall within.
    """

    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower_bound, upper_bound):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        self.lower_bound, self.upper_bound = utils.check_bounds(
            lower_bound, upper_bound
        )

    def bias(self, value):
        utils.check_numeric_value(value)
        shape = self.sensitivity / self.epsilon
        bias = shape * (
            np.exp((self.lower_bound + self.upper_bound - 2 * value) / shape) - 1
        )
        bias /= np.exp((self.lower_bound - value) / shape) + np.exp(
            (self.upper_bound - value) / shape
        )
        return bias

    def randomise(self, value):
        utils.check_numeric_value(value)
        noisy_value = super().randomise(value)
        return self._fold(noisy_value)

    def _fold(self, value):
        if value < self.lower_bound:
            return self._fold(2 * self.lower_bound - value)
        if value > self.upper_bound:
            return self._fold(2 * self.upper_bound - value)
        return value


class LaplaceBoundedDomain(LaplaceTruncated):
    r"""
    The bounded Laplace mechanism on a bounded domain.  The mechanism draws values directly from the domain using
    rejection sampling, without any post-processing [HABM20]_.

    References
    ----------
    .. [HABM20] Holohan, Naoise, Spiros Antonatos, Stefano Braghin, and Pól Mac Aonghusa. "The Bounded Laplace Mechanism
        in Differential Privacy." Journal of Privacy and Confidentiality 10, no. 1 (2020).

    """

    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower_bound, upper_bound):
        super().__init__(
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        self._rng = np.random.default_rng()

    def _find_scale(self):
        eps = self.epsilon
        delta = self.delta
        diam = self.upper_bound - self.lower_bound
        delta_q = self.sensitivity

        def _delta_c(shape):
            if shape == 0:
                return 2.0
            return (
                2 - np.exp(-delta_q / shape) - np.exp(-(diam - delta_q) / shape)
            ) / (1 - np.exp(-diam / shape))

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

        Returns:
        float: The effective :math:`\epsilon` parameter of the mechanism.  Returns ``None`` if `delta` is non-zero.

        """
        if self._scale is None:
            self._scale = self._find_scale()
        if self.delta > 0.0:
            return None
        return self.sensitivity / self._scale

    def bias(self, value):
        utils.check_numeric_value(value)

        if self._scale is None:
            self._scale = self._find_scale()

        bias = (self._scale - self.lower_bound + value) / 2 * np.exp(
            (self.lower_bound - value) / self._scale
        ) - (self._scale + self.upper_bound - value) / 2 * np.exp(
            (value - self.upper_bound) / self._scale
        )
        bias /= (
            1
            - np.exp((self.lower_bound - value) / self._scale) / 2
            - np.exp((value - self.upper_bound) / self._scale) / 2
        )
        return bias

    def variance(self, value):
        utils.check_numeric_value(value)
        if self._scale is None:
            self._scale = self._find_scale()
        variance = value**2
        variance -= (
            np.exp((self.lower_bound - value) / self._scale) * (self.lower_bound**2)
            + np.exp((value - self.upper_bound) / self._scale) * (self.upper_bound**2)
        ) / 2
        variance += self._scale * (
            self.lower_bound * np.exp((self.lower_bound - value) / self._scale)
            - self.upper_bound * np.exp((value - self.upper_bound) / self._scale)
        )
        variance += (self._scale**2) * (
            2
            - np.exp((self.lower_bound - value) / self._scale)
            - np.exp((value - self.upper_bound) / self._scale)
        )
        variance /= (
            1
            - (
                np.exp(-(value - self.lower_bound) / self._scale)
                + np.exp(-(self.upper_bound - value) / self._scale)
            )
            / 2
        )
        variance -= (self.bias(value) + value) ** 2
        return variance

    def randomise(self, value):
        utils.check_numeric_value(value)
        if self._scale is None:
            self._scale = self._find_scale()
        value = max(min(value, self.upper_bound), self.lower_bound)
        if np.isnan(value):
            return float("nan")
        samples = 1

        while True:
            noisy = value + self._scale * self._laplace_sampler(
                self._rng.random(samples),
                self._rng.random(samples),
                self._rng.random(samples),
                self._rng.random(samples),
            )
            if ((noisy >= self.lower_bound) & (noisy <= self.upper_bound)).any():
                idx = np.argmax(
                    (noisy >= self.lower_bound) & (noisy <= self.upper_bound)
                )
                return noisy[idx]
            samples = min(100000, samples * 2)


class LaplaceBoundedNoise(Laplace):
    r"""
    The Laplace mechanism with bounded noise, only applicable for approximate differential privacy (delta > 0)
    [GDGK18]_.

    Epsilon must be strictly positive, `epsilon` > 0. `delta` must be strictly in the interval (0, 0.5).
     - For zero `epsilon`, use :class:`.Uniform`.
     - For zero `delta`, use :class:`.Laplace`.

    .. [GDGK18] Geng, Quan, Wei Ding, Ruiqi Guo, and Sanjiv Kumar. "Truncated Laplacian Mechanism for Approximate
        Differential Privacy." arXiv preprint arXiv:1810.00877v1 (2018).

    """

    def __init__(self, *, epsilon, delta, sensitivity):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        if (
            epsilon == 0
        ):  # special requirements for epsilon and delta in LaplaceBoundedNoise
            raise ValueError(
                "Epsilon must be strictly positive. For zero epsilon, use :class:`.Uniform`."
            )
        if not 0 < delta < 0.5:
            raise ValueError(
                "Delta must be strictly in the interval (0,0.5). For zero delta, use :class:`.Laplace`."
            )
        self._noise_bound = None
        self._scale = None
        self._rng = np.random.default_rng()

    def bias(self, value):
        return 0.0

    def randomise(self, value):
        utils.check_numeric_value(value)
        if self._scale is None or self._noise_bound is None:
            self._scale = self.sensitivity / self.epsilon
            self._noise_bound = (
                0
                if self._scale == 0
                else self._scale
                * np.log(1 + (np.exp(self.epsilon) - 1) / 2 / self.delta)
            )
        if np.isnan(value):
            return float("nan")
        samples = 1
        while True:
            noisy = self._scale * self._laplace_sampler(
                self._rng.random(samples),
                self._rng.random(samples),
                self._rng.random(samples),
                self._rng.random(samples),
            )
            if ((noisy >= -self._noise_bound) & (noisy <= self._noise_bound)).any():
                idx = np.argmax(
                    (noisy >= -self._noise_bound) & (noisy <= self._noise_bound)
                )
                return value + noisy[idx]
            samples = min(100000, samples * 2)
