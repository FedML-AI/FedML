import numpy as np

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under local privacy" (ICML)


def GRR_Client(input_data, k, epsilon):
    """
    Generalized Randomized Response (GRR) protocol, a.k.a., direct encoding [1] or k-RR [2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: sanitized value.
    """

    if epsilon is not None or k is not None:
        
        # GRR parameters
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)

        # Mapping domain size k to the range [0, ..., k-1]
        domain = np.arange(k) 
        
        # GRR perturbation function
        rnd = np.random.random()
        if rnd <= p:
            return input_data

        else:
            return np.random.choice(domain[domain != input_data])

    else:
        raise ValueError('k (int) and epsilon (float) need a numerical value.')


def GRR_Aggregator(reports, k, epsilon):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all GRR-based sanitized values;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: normalized frequency (histogram) estimation.
    """

    if len(reports) == 0:

        raise ValueError('List of reports is empty.')
        
    else:

        if epsilon is not None or k is not None:
            
            # Number of reports
            n = len(reports)
            
            # GRR parameters
            p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
            q = (1 - p) / (k - 1)

            # Count how many times each value has been reported
            count_report = np.zeros(k)
            for rep in reports: 
                count_report[rep] += 1

            # Ensure non-negativity of estimated frequency
            est_freq = np.array((count_report - n*q) / (p-q)).clip(0)

            # Re-normalized estimated frequency
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))
                 
            return norm_est_freq

        else:
            raise ValueError('k (int) and epsilon (float) need a numerical value.')
