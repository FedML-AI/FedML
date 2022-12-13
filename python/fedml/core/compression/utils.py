
import logging
from copy import deepcopy
from datetime import datetime

import scipy.stats as stats


def gen_threshold_from_normal_distribution(p_value, mu, sigma):
    r"""PPF."""
    zvalue = stats.norm.ppf((1-p_value)/2)
    return mu+zvalue*sigma, mu-zvalue*sigma

