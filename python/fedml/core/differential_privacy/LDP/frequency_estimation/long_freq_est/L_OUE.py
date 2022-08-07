import numpy as np

# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).
# [2] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).

# The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from: https://github.com/hharcolezi/ldp-protocols-mobility-cdrs/blob/main/papers/%5B4%5D/1_ALLOMFREE_Analysis.ipynb

def L_OUE_Client(input_data, k, eps_perm, eps_1):

    """
    Longitudinal OUE (L-OUE) [1] protocol that chaines OUE [2] for both first and second rounds of sanitization.

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: sanitized UE vector.
    """
    if eps_1 >= eps_perm:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')
    
    else:
        # OUE parameters for round 1
        p1 = 0.5
        q1 = 1 / (np.exp(eps_perm) + 1)

        # OUE parameters for round 2
        p2 = 0.5
        q2 = ((-1.11803398874989 * np.sqrt(
            0.1 * np.exp(eps_1) + 0.05 * np.exp(2 * eps_1) + 0.3 * np.exp(eps_perm) + 0.45 * np.exp(2 * eps_perm) - np.exp(
                eps_1 + eps_perm) - 0.7 * np.exp(eps_1 + 2 * eps_perm) + 0.3 * np.exp(2 * eps_1 + eps_perm) + 0.45 * np.exp(
                2.0 * eps_1 + 2.0 * eps_perm) + 0.05) - 0.25 * np.exp(eps_1) - 0.25 * np.exp(eps_perm) + 0.75 * np.exp(
            eps_1 + eps_perm) - 0.25) * np.exp(-eps_perm)) / (np.exp(eps_1) - 1.0)

        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else:
            raise ValueError('Probabilities are negative, selecting eps_1 << eps_perm might probably solve it.')

        # Unary encoding
        input_ue_data = np.zeros(k)
        if input_data != None:
            input_ue_data[input_data] = 1

        # First round of sanitization (permanent memoization) with OUE using user's input_ue_data
        first_sanitization = np.zeros(k)
        for ind in range(k):
            if input_ue_data[ind] != 1:
                rnd = np.random.random()
                if rnd <= q1:
                    first_sanitization[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= p1:
                    first_sanitization[ind] = 1

        # Second round of sanitization with OUE using first_sanitization as input
        second_sanitization = np.zeros(k)
        for ind in range(k):
            if first_sanitization[ind] != 1:
                rnd = np.random.random()
                if rnd <= q2:
                    second_sanitization[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= p2:
                    second_sanitization[ind] = 1

        return second_sanitization

def L_OUE_Aggregator(ue_reports, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all L-OUE sanitized UE-based vectors;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation.
    """

    if len(ue_reports) == 0:
        raise ValueError('List of reports is empty.')

    # Number of reports
    n = len(ue_reports)

    # OUE parameters for round 1
    p1 = 0.5
    q1 = 1 / (np.exp(eps_perm) + 1)

    # OUE parameters for round 2
    p2 = 0.5
    q2 = ((-1.11803398874989 * np.sqrt(
        0.1 * np.exp(eps_1) + 0.05 * np.exp(2 * eps_1) + 0.3 * np.exp(eps_perm) + 0.45 * np.exp(2 * eps_perm) - np.exp(
            eps_1 + eps_perm) - 0.7 * np.exp(eps_1 + 2 * eps_perm) + 0.3 * np.exp(2 * eps_1 + eps_perm) + 0.45 * np.exp(
            2.0 * eps_1 + 2.0 * eps_perm) + 0.05) - 0.25 * np.exp(eps_1) - 0.25 * np.exp(eps_perm) + 0.75 * np.exp(
        eps_1 + eps_perm) - 0.25) * np.exp(-eps_perm)) / (np.exp(eps_1) - 1.0)

    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else:
        raise ValueError('Probabilities are negative.')

    # Ensure non-negativity of estimated frequency
    est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

    # Re-normalized estimated frequency
    if sum(est_freq) > 0:
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))
    
    else:
        norm_est_freq = est_freq

    return norm_est_freq
