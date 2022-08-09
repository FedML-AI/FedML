import numpy as np
from ..pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from ..pure_frequency_oracles.UE import UE_Client, UE_Aggregator
from ..pure_frequency_oracles.Variance_PURE import VAR_Pure

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).

def ADP_Client(input_data, k, epsilon, optimal=True):

    """
    Adaptive (ADP) protocol that minimizes variance value (i.e., either GRR or OUE from [1]).

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: sanitized value or UE vector.
    """

    if epsilon is not None or k is not None:

        # GRR parameters
        p_grr = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # Symmetric parameters (p+q = 1)
        p_ue = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        q_ue = 1 - p_ue

        # Optimized parameters
        if optimal:
            p_ue = 1 / 2
            q_ue = 1 / (np.exp(epsilon) + 1)

        # Variance values
        var_grr = VAR_Pure(p_grr, q_grr)
        var_ue = VAR_Pure(p_ue, q_ue)

        # Adaptive protocol
        if var_grr <= var_ue:

            return GRR_Client(input_data, k, epsilon)
        else:

            return UE_Client(input_data, k, epsilon, optimal)

    else:
        raise ValueError('k (int) and epsilon (float) need a numerical value.')


def ADP_Aggregator(reports, k, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all GRR sanitized values or UE-based vectors;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: normalized frequency (histogram) estimation.
    """

    if len(reports) == 0:

        raise ValueError('List of reports is empty.')

    else:

        if epsilon is not None or k is not None:

            # GRR parameters
            p_grr = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
            q_grr = (1 - p_grr) / (k - 1)

            # Symmetric parameters (p+q = 1)
            p_ue = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
            q_ue = 1 - p_ue

            # Optimized parameters
            if optimal:
                p_ue = 1 / 2
                q_ue = 1 / (np.exp(epsilon) + 1)

            # Variance values
            var_grr = VAR_Pure(p_grr, q_grr)
            var_ue = VAR_Pure(p_ue, q_ue)

            # Adaptive estimator
            if var_grr <= var_ue:

                return GRR_Aggregator(reports, k, epsilon)
            else:

                return UE_Aggregator(reports, epsilon, optimal)

        else:
            raise ValueError('k (int) and epsilon (float) need a numerical value.')
