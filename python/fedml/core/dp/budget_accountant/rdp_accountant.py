"""
Reference:
autodp: https://github.com/yuxiangw/autodp
"""
import numpy as np


def stable_logsumexp_two(x, y):
    a = np.maximum(x, y)
    # if np.isneginf(a):
    #    return a
    # else:
    return a + np.log(np.exp(x - a) + np.exp(y - a))


class RDP_Accountant:
    def __init__(self, dp_param, alpha, dp_mechanism="gaussian"):
        self.alpha = alpha
        if dp_mechanism == "gaussian":
            self.rdp_epsilon = self.RDP_gaussian(sigma=dp_param)
        elif dp_mechanism == "laplace":
            self.rdp_epsilon = self.RDP_laplace(rdp_scale=dp_param)
        else:
            raise Exception(f"the DP mechanism is not supported: {dp_mechanism}")

    def RDP_gaussian(self, sigma):
        """
        Args:
            sigma: normalized noise level: std divided by global L2 sensitivity
            alpha: The order of the Renyi Divergence

        Return: Evaluation of the RDP's epsilon
        """
        assert (sigma > 0)
        assert (self.alpha >= 0)
        return 0.5 / sigma ** 2 * self.alpha

    def RDP_laplace(self, rdp_scale):
        """
        Args:
            rdp_scale: the ratio of the scale parameter and L1 sensitivity
            alpha: The order of the Renyi Divergence
        Return: Evaluation of the RDP's epsilon
        """
        alpha = 1.0 * self.alpha
        if np.isinf(alpha):
            return 1 / rdp_scale
        elif alpha == 1:
            # KL-divergence
            return 1 / rdp_scale + np.exp(-1 / rdp_scale) - 1
        elif alpha > 1:  # alpha > 1
            return stable_logsumexp_two((alpha - 1.0) / rdp_scale + np.log(alpha / (2.0 * alpha - 1)),
                                        -1.0 * alpha / rdp_scale + np.log((alpha - 1.0) / (2.0 * alpha - 1))) / (alpha - 1)
        elif alpha == 0.5:
            return -2 * (-1.0 / (2 * rdp_scale) + np.log(1 + 1.0 / (2 * rdp_scale)))  # -2*np.log(np.exp(-1.0/(2*b))*(1+1.0/(2*b)))
        else:
            return np.log(
                alpha / (2.0 * alpha - 1) * np.exp((alpha - 1.0) / rdp_scale) + (alpha - 1.0) / (2.0 * alpha - 1) * np.exp(
                    -1.0 * alpha / rdp_scale)) / (alpha - 1)
            # Handling the case when alpha = 1/2?

    def get_accountant(self):
        return self.rdp_epsilon


