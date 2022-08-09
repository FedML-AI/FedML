import numpy as np
from ..pure_frequency_oracles.GRR import GRR_Client
from ..pure_frequency_oracles.UE import UE_Client
from ..mdim_freq_est.Variance_RSpFD import VAR_RSpFD_GRR, VAR_RSpFD_UE_zero

# [1] Arcolezi et al (2021) "Random Sampling Plus Fake Data: Multidimensional Frequency Estimates With Local Differential Privacy" (ACM CIKM).
# [2] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).

def RSpFD_GRR_Client(input_tuple, lst_k, d, epsilon):

    """
    Random Sampling Plus Fake Data (RS+FD) with Generalized Randomized Response (GRR) protocol as local randomizer [1].

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :return: tuple of sanitized/fake values.
    """

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # RS+FD perturbation function
    sanitized_tuple = []
    for idx in range(d):

        # Fake data generation
        if idx != rnd_att:
            sanitized_tuple.append(np.random.randint(lst_k[idx]))

        # Local randomization
        else:
            sanitized_tuple.append(GRR_Client(input_tuple[idx], lst_k[idx], amp_eps))

    return sanitized_tuple

def RSpFD_UE_zero_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    """
    Random Sampling Plus Fake Data (RS+FD) with Unary Encoding (UE) protocol as local randomizer [1].

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: tuple of sanitized/fake UE vectors.
    """

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # RS+FD perturbation function
    sanitized_ue_tuple = []
    for idx in range(d):

        # Fake data generation (with UE applied to a zero vector)
        if idx != rnd_att:

            sanitized_ue_tuple.append(UE_Client(None, lst_k[idx], amp_eps, optimal))

        # Local randomization
        else:
            sanitized_ue_tuple.append(UE_Client(input_tuple[idx], lst_k[idx], amp_eps, optimal))

    return sanitized_ue_tuple

def RSpFD_UE_rnd_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    """
    Random Sampling Plus Fake Data (RS+FD) with Unary Encoding (UE) protocol as local randomizer [1].

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: tuple of sanitized/fake UE vectors.
    """

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # RS+FD perturbation function
    sanitized_ue_tuple = []
    for idx in range(d):

        # Fake data generation (with UE applied to a random UE vector)
        if idx != rnd_att:
            sanitized_ue_tuple.append(UE_Client(np.random.randint(lst_k[idx]), lst_k[idx], amp_eps, optimal))

        # Local randomization
        else:
            sanitized_ue_tuple.append(UE_Client(input_tuple[idx], lst_k[idx], amp_eps, optimal))

    return sanitized_ue_tuple

def RSpFD_ADP_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    """
    Random Sampling Plus Fake Data (RS+FD) with Adaptive (ADP) protocol (i.e., either RS+FD[GRR] or RS+FD[OUE-z] [1]).

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: tuple of sanitized/fake values or UE vectors.
    """

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # RS+FD perturbation function
    sanitized_tuple = []
    for idx in range(d):
        k = lst_k[idx]

        # GRR parameters with amplified epsilon
        p_grr = np.exp(amp_eps) / (np.exp(amp_eps) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # UE parameters with amplified epsilon
        p_ue = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q_ue = 1 - p_ue

        if optimal:
            p_ue = 0.5
            q_ue = 1 / (np.exp(amp_eps) + 1)

        # variance values of using RS+FD[GRR] and RS+FD[OUE-z]
        var_grr = VAR_RSpFD_GRR(p_grr, q_grr, k, d)
        var_ue = VAR_RSpFD_UE_zero(p_ue, q_ue, d)

        # Fake data generation (with either uniform at random or with UE applied to a zero vector)
        if idx != rnd_att:

            # Adaptive fake data generation
            if var_grr <= var_ue:
                sanitized_tuple.append(np.random.randint(k))

            else:
                sanitized_tuple.append(UE_Client(None, k, amp_eps, optimal))

        # Local randomization
        else:

            # Adaptive protocol
            if var_grr <= var_ue:
                sanitized_tuple.append(GRR_Client(input_tuple[idx], k, amp_eps))

            else:
                sanitized_tuple.append(UE_Client(input_tuple[idx], k, amp_eps, optimal))

    return sanitized_tuple

def RSpFD_GRR_Aggregator(reports_tuple, lst_k, d, epsilon):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all RS+FD[GRR]-based sanitized tuples;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Number of reports
    n = len(reports_tuple)

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        # GRR parameters with amplified epsilon
        p = np.exp(amp_eps) / (np.exp(amp_eps) + k - 1)
        q = (1 - p) / (k - 1)

        # Count how many times each value has been reported
        count_report = np.zeros(k)
        for rep in reports:
            count_report[rep] += 1

        # Ensure non-negativity of estimated frequency
        est_freq = np.array(((count_report * d * k) - n * (d - 1 + q * k)) / (n * k * (p - q))).clip(0)

        # Re-normalized estimated frequency
        norm_est_freq = est_freq / sum(est_freq)

        lst_freq_est.append(norm_est_freq)

    return np.array(lst_freq_est, dtype='object')

def RSpFD_UE_zero_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all RS+FD[UE-zero]-based sanitized tuples;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Number of reports
    n = len(reports_tuple)

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        # UE parameters with amplified epsilon
        p = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q = 1 - p

        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(amp_eps) + 1)

        # Ensure non-negativity of estimated frequency
        est_freq = np.array(d*(sum(reports) - n * q) / (n * (p - q))).clip(0)

        # Re-normalized estimated frequency
        norm_est_freq = est_freq / sum(est_freq)

        lst_freq_est.append(norm_est_freq)

    return np.array(lst_freq_est, dtype='object')

def RSpFD_UE_rnd_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all RS+FD[UE-rnd]-based sanitized tuples;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Number of reports
    n = len(reports_tuple)

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        # UE parameters with amplified epsilon
        p = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q = 1 - p

        if optimal:
            p = 1 / 2
            q = 1 / (np.exp(amp_eps) + 1)

        # Ensure non-negativity of estimated frequency
        est_freq = np.array(((sum(reports) * d * k) - n * (q * k + (p - q) * (d - 1) + q * k * (d - 1))) / (n * k * (p - q))).clip(0)

        # Re-normalized estimated frequency
        norm_est_freq = est_freq / sum(est_freq)

        lst_freq_est.append(norm_est_freq)

    return np.array(lst_freq_est, dtype='object')

def RSpFD_ADP_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all RS+FD[GRR]- / RS+FD[UE-zero]-based sanitized tuples;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Number of reports
    n = len(reports_tuple)

    # Amplified epsilon parameter
    amp_eps = np.log(d * (np.exp(epsilon) - 1) + 1)

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        k = lst_k[idx]

        # GRR parameters with amplified epsilon
        p_grr = np.exp(amp_eps) / (np.exp(amp_eps) + k - 1)
        q_grr = (1 - p_grr) / (k - 1)

        # UE parameters with amplified epsilon
        p_ue = np.exp(amp_eps / 2) / (np.exp(amp_eps / 2) + 1)
        q_ue = 1 - p_ue

        if optimal:
            p_ue = 0.5
            q_ue = 1 / (np.exp(amp_eps) + 1)

        # Variance values of using RS+FD[GRR] or RS+FD[OUE-z]
        var_grr = VAR_RSpFD_GRR(p_grr, q_grr, k, d)
        var_ue = VAR_RSpFD_UE_zero(p_ue, q_ue, d)

        # Adaptive estimator
        if var_grr <= var_ue:

            # Count how many times each value has been reported
            count_report = np.zeros(k)
            for rep in reports:
                count_report[rep] += 1

            # Ensure non-negativity of estimated frequency
            est_freq = np.array(((count_report * d * k) - n * (d - 1 + q_grr * k)) / (n * k * (p_grr - q_grr))).clip(0)

        else:
            # Ensure non-negativity of estimated frequency
            est_freq = np.array(d * (sum(reports) - n * q_ue) / (n * (p_ue - q_ue))).clip(0)

        # Re-normalized estimated frequency
        norm_est_freq = est_freq / sum(est_freq)

        lst_freq_est.append(norm_est_freq)

    return np.array(lst_freq_est, dtype='object')
