import numpy as np
from ..long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator
from ..long_freq_est.L_OUE import L_OUE_Client, L_OUE_Aggregator
from ..long_freq_est.L_OSUE import L_OSUE_Client, L_OSUE_Aggregator
from ..long_freq_est.L_SUE import L_SUE_Client, L_SUE_Aggregator
from ..long_freq_est.L_SOUE import L_SOUE_Client, L_SOUE_Aggregator
from ..long_freq_est.L_ADP import L_ADP_Client, L_ADP_Aggregator
from ..long_freq_est.dBitFlipPM import dBitFlipPM_Client, dBitFlipPM_Aggregator

# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).
# [2] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [3] Ding, Kulkarni, and Yekhanin (2017) "Collecting telemetry data privately." (NeurIPS).

def SMP_L_GRR_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Sampling (SMP) a single attribute and using L-GRR [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple (sampled attribute, sanitized value).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with L-GRR protocol
    att_sanitized_value = (rnd_att, L_GRR_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_OUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Sampling (SMP) a single attribute and using L-OUE [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple (sampled attribute, sanitized UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with L-OUE protocol
    att_sanitized_value = (rnd_att, L_OUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_OSUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Sampling (SMP) a single attribute and using L-OSUE [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple (sampled attribute, sanitized UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with L-OSUE protocol
    att_sanitized_value = (rnd_att, L_OSUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_SUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Sampling (SMP) a single attribute and using L-SUE [1] (a.k.a. Basic RAPPOR [2]) protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple (sampled attribute, sanitized UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with L-SUE protocol
    att_sanitized_value = (rnd_att, L_SUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_SOUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Sampling (SMP) a single attribute and using L-SOUE [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple (sampled attribute, sanitized UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with L-SOUE protocol
    att_sanitized_value = (rnd_att, L_SOUE_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value
    
def SMP_dBitFlipPM_Client(input_tuple, lst_k, lst_b, d_bits, d, eps_perm):

    """
    Sampling (SMP) a single attribute and using dBitFlipPM [3] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param lst_b: list of attributes' new domain size (bucketized);
    :param d_bits: number of bits to report;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :return: tuple (sampled attribute, sanitized UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with L-GRR protocol
    att_sanitized_value = (rnd_att, dBitFlipPM_Client(input_tuple[rnd_att], lst_k[rnd_att], lst_b[rnd_att], d_bits, eps_perm))

    return att_sanitized_value    

def SMP_L_ADP_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Sampling (SMP) a single attribute and using L-ADP [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple (sampled attribute, sanitized value or UE vector).
    """

    # Select an attribute at random
    rnd_att = np.random.randint(d)

    # Report the sampled attribute and its LDP value with L-ADP protocol
    att_sanitized_value = (rnd_att, L_ADP_Client(input_tuple[rnd_att], lst_k[rnd_att], eps_perm, eps_1))

    return att_sanitized_value

def SMP_L_GRR_Aggregator(reports_tuple, lst_k, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all tuples (sampled attribute, sanitized value);
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
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
        lst_freq_est.append(L_GRR_Aggregator(dic_rep_smp[idx], lst_k[idx], eps_perm, eps_1))

    return np.array(lst_freq_est, dtype='object')

def SMP_L_OUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_ue_tuple: list of all tuples (sampled attribute, sanitized UE vector);
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_ue_tuple) == 0:
        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_OUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return np.array(lst_freq_est, dtype='object')

def SMP_L_OSUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_ue_tuple: list of all tuples (sampled attribute, sanitized UE vector);
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_ue_tuple) == 0:
        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_OSUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return np.array(lst_freq_est, dtype='object')

def SMP_L_SUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_ue_tuple: list of all tuples (sampled attribute, sanitized UE vector);
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_ue_tuple) == 0:
        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_SUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return np.array(lst_freq_est, dtype='object')

def SMP_L_SOUE_Aggregator(reports_ue_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_ue_tuple: list of all tuples (sampled attribute, sanitized UE vector);
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_ue_tuple) == 0:
        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_SOUE_Aggregator(dic_rep_smp[idx], eps_perm, eps_1))

    return np.array(lst_freq_est, dtype='object')
    
def SMP_dBitFlipPM_Aggregator(reports_ue_tuple, lst_b, d_bits, d, eps_perm):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_ue_tuple: list of all tuples (sampled attribute, sanitized UE vector);
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_ue_tuple) == 0:
        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(dBitFlipPM_Aggregator(dic_rep_smp[idx], lst_b[idx], d_bits, eps_perm))

    return np.array(lst_freq_est, dtype='object')    

def SMP_L_ADP_Aggregator(reports_ue_tuple, lst_k, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_ue_tuple: list of all tuples (sampled attribute, sanitized value or UE vector);
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    if len(reports_ue_tuple) == 0:
        raise ValueError('List of reports is empty.')

    # Gather users reporting the same attribute
    dic_rep_smp = {att: [] for att in range(d)}
    for val in reports_ue_tuple:
        dic_rep_smp[val[0]].append(val[-1])

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        lst_freq_est.append(L_ADP_Aggregator(dic_rep_smp[idx], lst_k[idx], eps_perm, eps_1))

    return np.array(lst_freq_est, dtype='object')
