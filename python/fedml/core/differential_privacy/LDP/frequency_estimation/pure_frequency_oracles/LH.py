import numpy as np
from sys import maxsize
import xxhash

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Bassily and Smith "Local, private, efficient protocols for succinct histograms" (STOC).

# Code adapted from pure-ldp library (https://github.com/Samuel-Maddock/pure-LDP) developed by Samuel Maddock

def LH_Client(input_data, epsilon, optimal=True):
    
    """
    Local Hashing (LH) protocol[1], which is logically equivalent to the random matrix projection technique in [2].

    :param input_data: user's true value;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized LH (OLH) protocol from [1];
    :return: tuple of sanitized value and random seed.
    """
    
    # Binary LH (BLH) parameter
    g = 2
    
    # Optimal LH (OLH) parameter
    if optimal:
        g = int(round(np.exp(epsilon))) + 1
    
    # GRR parameters with reduced domain size g
    p = np.exp(epsilon) / (np.exp(epsilon) + g - 1)
    q = 1 / (np.exp(epsilon) + g - 1)
    
    # Generate random seed and hash the user's value
    rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
    hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % g)
    
    # LH perturbation function (i.e., GRR-based)
    sanitized_value = hashed_input_data
    rnd = np.random.random()
    if rnd > p - q:
        
        sanitized_value = np.random.randint(0, g)
        
    return (sanitized_value, rnd_seed)

def LH_Aggregator(reports, k, epsilon, optimal=True):
    
    if len(reports) == 0:

        raise ValueError('List of reports is empty.')
        
    else:

        if epsilon is not None or k is not None:
            
            # Number of reports
            n = len(reports)
            
            # Binary LH (BLH) parameter
            g = 2

            # Optimal LH (OLH) parameter
            if optimal:
                g = int(round(np.exp(epsilon))) + 1

            # GRR parameters with reduced domain size g
            p = np.exp(epsilon) / (np.exp(epsilon) + g - 1)
            q = 1 / (np.exp(epsilon) + g - 1)

            # Count how many times each value has been reported
            count_report = np.zeros(k)
            for tuple_val_seed in reports:
                for v in range(k):
                    if tuple_val_seed[0] == (xxhash.xxh32(str(v), seed=tuple_val_seed[1]).intdigest() % g):
                        count_report[v] += 1
            
            # Ensure non-negativity of estimated frequency
            a = g / (p * g - 1)
            b = n / (p * g - 1)
            est_freq = (a * count_report - b).clip(0)

            # Re-normalized estimated frequency
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))
                 
            return norm_est_freq
       
        else:
            raise ValueError('k (int) and epsilon (float) need a numerical value.')