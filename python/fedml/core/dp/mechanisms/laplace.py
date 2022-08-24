import secrets
import numpy as np
from ..common.utils import check_numeric_value, check_params


class Laplace:
    """
    The classical Laplace mechanism in differential privacy.
    This code refers to IBM DP Library: https://github.com/IBM/differential-privacy-library
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
        check_params(epsilon, delta, sensitivity)
        self.scale = float(sensitivity) / (float(epsilon) - np.log(1 - float(delta)))
        self._rng = secrets.SystemRandom()

    def bias(self, value):
        """Returns the bias of the mechanism at a given `value`."""
        return 0.0

    def variance(self, value):
        """Returns the variance of the mechanism at a given `value`."""
        return 2 * self.scale ** 2

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(
            np.pi * unif4
        )

    def randomise(self, value):
        """Randomise `value` with the mechanism."""
        check_numeric_value(value)
        noise = self.compute_a_noise()
        return value + noise

    def compute_a_noise(self):

        standard_laplace = self._laplace_sampler(
            self._rng.random(),
            self._rng.random(),
            self._rng.random(),
            self._rng.random(),
        )
        return -self.scale * standard_laplace

