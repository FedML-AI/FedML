import numpy as np

# [1] Ye and Barg (2018) "Optimal schemes for discrete distribution estimation under locally differential privacy" (IEEE Transactions on Information Theory)
# [2] Wang et al (2016) "Mutual information optimally local private discrete distribution estimation" (arXiv:1607.08025).


def SS_Client(input_data, k, epsilon):
    """
    Subset Selection (SS) protocol [1,2].

    :param input_data: user's true value;
    :param k: attribute's domain size;
    :param epsilon: privacy guarantee;
    :return: set of sub_k sanitized values.
    """
    
    # Mapping domain size k to the range [0, ..., k-1]
    domain = np.arange(k) 
    
    # SS parameters
    sub_k = int(max(1, np.rint(k / (np.exp(epsilon) + 1))))
    p_v = sub_k * np.exp(epsilon) / (sub_k * np.exp(epsilon) + k - sub_k)
    
    # SS perturbation function        
    rnd = np.random.random()    
    sub_set = np.zeros(sub_k, dtype='int64')
    if rnd <= p_v:
        sub_set[0] = int(input_data)
        sub_set[1:] = np.random.choice(domain[domain != input_data], size=sub_k-1, replace=False)
        return sub_set
        
    else:
        return np.random.choice(domain[domain != input_data], size=sub_k, replace=False) 

def SS_Aggregator(reports, k, epsilon):
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

            # SS parameters    
            sub_k = int(max(1, np.rint(k / (np.exp(epsilon) + 1))))
            p = sub_k * np.exp(epsilon) / (sub_k * np.exp(epsilon) + k - sub_k)
            q = ((sub_k - 1) * (sub_k * np.exp(epsilon)) + (k - sub_k) * sub_k) / ((k - 1) * (sub_k * np.exp(epsilon) + k - sub_k))

            # Count how many times each value has been reported
            count_report = np.zeros(k)
            for rep in reports:
                for i in range(sub_k):
                    count_report[rep[i]] += 1

            # Ensure non-negativity of estimated frequency
            est_freq = np.array((count_report - n*q) / (p-q)).clip(0)

            # Re-normalized estimated frequency
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

            return norm_est_freq
        
        else:
            raise ValueError('k (int) and epsilon (float) need a numerical value.')