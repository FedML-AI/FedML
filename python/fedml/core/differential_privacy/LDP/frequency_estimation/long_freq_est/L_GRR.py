import numpy as np
from ..pure_frequency_oracles.GRR import GRR_Client

# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).
# [2] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under local privacy" (ICML)

# The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from: https://github.com/hharcolezi/ldp-protocols-mobility-cdrs/blob/main/papers/%5B4%5D/1_ALLOMFREE_Analysis.ipynb

def L_GRR_Client(input_data, k, eps_perm, eps_1):

    """
    Longitudinal GRR (L-GRR) [1] protocol that chaines GRR [2] for both first and second rounds of sanitization.

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: sanitized value.
    """
    
    if eps_1 >= eps_perm:
        raise ValueError('Please set eps_1 (single report, i.e., lower bound) < eps_perm (infinity reports, i.e., upper bound)')
    
    else:
        # GRR parameters for round 1
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1 = (1 - p1) / (k - 1)

        # GRR parameters for round 2
        p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(k-1)+q1)
        q2 = (1 - p2) / (k-1)
        
        if (np.array([p1, q1, p2, q2]) >= 0).all():
            pass
        else: 
            raise ValueError('Probabilities are negative, selecting eps_1 << eps_perm might probably solve it.')

        # Get epsilon of second round of sanitization
        eps_sec_round = np.log(p2 / q2)

        # First round of sanitization (permanent memoization) with GRR using user's input_data
        first_sanitization = GRR_Client(input_data, k, eps_perm)

        # Second round of sanitization with GRR using first_sanitization as input
        second_sanitization = GRR_Client(first_sanitization, k, eps_sec_round)
        
        return second_sanitization

def L_GRR_Aggregator(reports, k, eps_perm, eps_1):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all L-GRR sanitized values;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation.
    """

    if len(reports) == 0:
        raise ValueError('List of reports is empty.')

    # Number of reports
    n = len(reports)
                
    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    # GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(k-1)+q1)
    q2 = (1 - p2) / (k-1)
    
    if (np.array([p1, q1, p2, q2]) >= 0).all():
        pass
    else: 
        raise ValueError('Probabilities are negative.')

    # Count how many times each value has been reported
    count_report = np.zeros(k)            
    for rep in reports:
        count_report[rep] += 1

    # Ensure non-negativity of estimated frequency
    est_freq = ((count_report - n*q1*(p2-q2) - n*q2) / (n*(p1-q1)*(p2-q2))).clip(0)

    # Re-normalized estimated frequency
    if sum(est_freq) > 0:
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))
    
    else:
        norm_est_freq = est_freq

    return norm_est_freq
