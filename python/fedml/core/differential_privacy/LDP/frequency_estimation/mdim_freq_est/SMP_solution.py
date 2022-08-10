import numpy as np
from ..pure_frequency_oracles.GRR import GRR_Client, GRR_Aggregator
from ..pure_frequency_oracles.UE import UE_Client, UE_Aggregator
from ..pure_frequency_oracles.ADP import ADP_Client, ADP_Aggregator
from ..pure_frequency_oracles.LH import LH_Client, LH_Aggregator
from ..pure_frequency_oracles.SS import SS_Client, SS_Aggregator

# [1] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under local privacy" (ICML)
# [2] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [3] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [4] Ye and Barg (2018) "Optimal schemes for discrete distribution estimation under locally differential privacy" (IEEE Transactions on Information Theory)
# [5] Wang et al (2016) "Mutual information optimally local private discrete distribution estimation" (arXiv:1607.08025).

def SMP_GRR_Client(input_tuple, lst_k, d, epsilon):

    """
    Sampling (SMP) a single attribute and using Generalized Randomized Response (GRR) [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :return: tuple (sampled attribute, sanitized value).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with GRR protocol
    att_sanitized_value = (rnd_att, GRR_Client(input_tuple[rnd_att], lst_k[rnd_att], epsilon))

    return att_sanitized_value

def SMP_UE_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    """
    Sampling (SMP) a single attribute and using Unary Encoding (UE) protocol [2] as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
    :return: tuple (sampled attribute, sanitized UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with UE protocol
    att_sanitized_value = (rnd_att, UE_Client(input_tuple[rnd_att], lst_k[rnd_att], epsilon, optimal))
    
    return att_sanitized_value

def SMP_LH_Client(input_tuple, d, epsilon, optimal=True):

    """
    Sampling (SMP) a single attribute and using Local Hashing (LH) protocol [3] as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized LH (OLH) protocol from [3];
    :return: tuple (sampled attribute and tuple of sanitized value and random seed).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with UE protocol
    att_sanitized_value = (rnd_att, LH_Client(input_tuple[rnd_att], epsilon, optimal))
    
    return att_sanitized_value

def SMP_SS_Client(input_tuple, lst_k, d, epsilon):

    """
    Sampling (SMP) a single attribute and using Subset Selection (SS) [4,5] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :return: tuple (sampled attribute, sanitized value).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with GRR protocol
    att_sanitized_value = (rnd_att, SS_Client(input_tuple[rnd_att], lst_k[rnd_att], epsilon))

    return att_sanitized_value

def SMP_ADP_Client(input_tuple, lst_k, d, epsilon, optimal=True):

    """
    Sampling (SMP) a single attribute and using Adaptive (ADP) protocol (i.e., GRR or UE) as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
    :return: tuple (sampled attribute, sanitized value or UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with ADP protocol
    att_sanitized_value = (rnd_att, ADP_Client(input_tuple[rnd_att], lst_k[rnd_att], epsilon, optimal))

    return att_sanitized_value

def SMP_GRR_Aggregator(reports_tuple, lst_k, d, epsilon):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all tuples (sampled attribute, sanitized value);
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att:[] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(GRR_Aggregator(dic_rep_smp[idx], lst_k[idx], epsilon))

    return np.array(lst_freq_est, dtype='object')

def SMP_UE_Aggregator(reports_tuple, d, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all tuples (sampled attribute, sanitized UE vector);
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att:[] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(UE_Aggregator(dic_rep_smp[idx], epsilon, optimal))

    return np.array(lst_freq_est, dtype='object')

def SMP_LH_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all tuples (sampled attribute, sanitized value, and random seed);
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized LH (OLH) protocol from [3];
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att:[] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(LH_Aggregator(dic_rep_smp[idx], lst_k[idx], epsilon, optimal))

    return np.array(lst_freq_est, dtype='object')
    
def SMP_SS_Aggregator(reports_tuple, lst_k, d, epsilon):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all tuples (sampled attribute, sanitized value);
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:

        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att:[] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(SS_Aggregator(dic_rep_smp[idx], lst_k[idx], epsilon))

    return np.array(lst_freq_est, dtype='object')    

def SMP_ADP_Aggregator(reports_tuple, lst_k, d, epsilon, optimal=True):
    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all tuples (sampled attribute, sanitized value or UE vector);
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param epsilon: privacy guarantee;
    :param optimal: if True, it uses the Optimized UE (OUE) protocol from [3];
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_tuple) == 0:
        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):

        lst_freq_est.append(ADP_Aggregator(dic_rep_smp[idx], lst_k[idx], epsilon, optimal))

    return np.array(lst_freq_est, dtype='object')
