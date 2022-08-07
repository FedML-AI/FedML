import numpy as np

# [1] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [2] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).


def UE_Client(input_data, k, epsilon, optimal=True):
    """
    Unary Encoding (UE) protocol, a.k.a. Basic One-Time RAPPOR (if optimal=False) [1]

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: sanitized UE vector.
    """

    # Symmetric parameters (p+q = 1)
    p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
    q = 1 - p

    # Optimized parameters
    if optimal:
        p = 1 / 2
        q = 1 / (np.exp(epsilon) + 1)
        
    # Unary encoding
    input_ue_data = np.zeros(k)
    if input_data != None:
        input_ue_data[input_data] = 1

    # Initializing a zero-vector
    sanitized_vec = np.zeros(k)

    # UE perturbation function
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q:
                sanitized_vec[ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p:
                sanitized_vec[ind] = 1
    return sanitized_vec
        
def UE_Aggregator(reports, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

    :param reports: list of all UE-based sanitized vectors;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: normalized frequency (histogram) estimation.
    """

    if len(reports) == 0: 
        
        raise ValueError('List of reports is empty.')
        
    else:

        if epsilon is not None:

            # Number of reports
            n = len(reports)

            # Symmetric parameters (p+q = 1)
            p = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
            q = 1 - p

            # Optimized parameters
            if optimal:
                p = 1 / 2
                q = 1 / (np.exp(epsilon) + 1)

            # Ensure non-negativity of estimated frequency
            est_freq = np.array((sum(reports) - q * n) / (p-q)).clip(0)

            # Re-normalized estimated frequency
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))
                 
            return norm_est_freq

        else:
            raise ValueError('epsilon (float) needs a numerical value.')
