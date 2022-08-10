import numpy as np
from ...long_freq_est.L_GRR import L_GRR_Client, L_GRR_Aggregator
from ...long_freq_est.L_OUE import L_OUE_Client, L_OUE_Aggregator
from ...long_freq_est.L_OSUE import L_OSUE_Client, L_OSUE_Aggregator
from ...long_freq_est.L_SUE import L_SUE_Client, L_SUE_Aggregator
from ...long_freq_est.L_SOUE import L_SOUE_Client, L_SOUE_Aggregator
from ...long_freq_est.L_ADP import L_ADP_Client, L_ADP_Aggregator
from ...long_freq_est.dBitFlipPM import dBitFlipPM_Client, dBitFlipPM_Aggregator

# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).
# [2] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [3] Ding, Kulkarni, and Yekhanin (2017) "Collecting telemetry data privately." (NeurIPS).

def SPL_L_GRR_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Splitting (SPL) the privacy budget and using L-GRR [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple of sanitized values.
    """

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Sanitization of each value with L-GRR protocol
    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_GRR_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_OUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    """
    Splitting (SPL) the privacy budget and using L-OUE [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple of sanitized UE vectors.
    """

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Sanitization of each value with L-OUE protocol
    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_OUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_OSUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):
    """
    Splitting (SPL) the privacy budget and using L-OSUE [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple of sanitized UE vectors.
    """

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Sanitization of each value with L-OSUE protocol
    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_OSUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_SUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Splitting (SPL) the privacy budget and using L-SUE [1] (a.k.a. Basic RAPPOR [2]) protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple of sanitized UE vectors.
    """

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Sanitization of each value with L-SUE protocol
    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_SUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_SOUE_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Splitting (SPL) the privacy budget and using L-SOUE [1] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple of sanitized UE vectors.
    """

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Sanitization of each value with L-SOUE protocol
    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_SOUE_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple
    
def SPL_dBitFlipPM_Client(input_tuple, lst_k, lst_b, d_bits, d, eps_perm):

    """
    Splitting (SPL) the privacy budget and using dBitFlipPM [3] protocol as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param lst_b: list of attributes' new domain size (bucketized);
    :param d_bits: number of bits to report;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :return: tuple of sanitized UE vectors.
    """

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d

    # Sanitization of each value with L-GRR protocol
    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(dBitFlipPM_Client(input_tuple[idx], lst_k[idx], lst_b[idx], d_bits, eps_perm_spl))

    return sanitized_tuple    

def SPL_L_ADP_Client(input_tuple, lst_k, d, eps_perm, eps_1):

    """
    Splitting (SPL) the privacy budget and using L-ADP [1] protocol (a.k.a. ALLOMFREE [1]) as local randomizer.

    :param input_tuple: user's true tuple of values;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: tuple of sanitized UE vectors.
    """

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Sanitization of each value with L-ADP protocol
    sanitized_tuple = []
    for idx in range(d):
        sanitized_tuple.append(L_ADP_Client(input_tuple[idx], lst_k[idx], eps_perm_spl, eps_1_spl))

    return sanitized_tuple

def SPL_L_GRR_Aggregator(reports_tuple, lst_k, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all sanitized tuples;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_GRR_Aggregator(reports, lst_k[idx], eps_perm_spl, eps_1_spl))

    return np.array(lst_freq_est, dtype='object')

def SPL_L_OUE_Aggregator(reports_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all sanitized tuples;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_OUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return np.array(lst_freq_est, dtype='object')

def SPL_L_OSUE_Aggregator(reports_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all sanitized tuples;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_OSUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return np.array(lst_freq_est, dtype='object')
    
def SPL_dBitFlipPM_Aggregator(reports_tuple, lst_b, d_bits, d, eps_perm):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all sanitized tuples;
    :param lst_b: list of attributes' new domain size (bucketized);
    :param d_bits: number of bits to report;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(dBitFlipPM_Aggregator(reports, lst_b[idx], d_bits, eps_perm_spl))

    return np.array(lst_freq_est, dtype='object')

def SPL_L_SUE_Aggregator(reports_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all sanitized tuples;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_SUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return np.array(lst_freq_est, dtype='object')

def SPL_L_SOUE_Aggregator(reports_tuple, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all sanitized tuples;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_SOUE_Aggregator(reports, eps_perm_spl, eps_1_spl))

    return np.array(lst_freq_est, dtype='object')

def SPL_L_ADP_Aggregator(reports_tuple, lst_k, d, eps_perm, eps_1):

    """
    Statistical Estimator for Normalized Frequency (0 -- 1) of all d attributes with post-processing to ensure non-negativity.

    :param reports_tuple: list of all sanitized tuples;
    :param lst_k: list of attributes' domain size;
    :param d: number of attributes;
    :param eps_perm: upper bound of privacy guarantee (infinity reports);
    :param eps_1: lower bound of privacy guarantee (a single report), thus, eps_1 < eps_perm;
    :return: normalized frequency (histogram) estimation of all d attributes.
    """

    reports_tuple = np.array(reports_tuple, dtype='object')

    # Splitting the privacy budget over the number of attributes d
    eps_perm_spl = eps_perm / d
    eps_1_spl = eps_1 / d

    # Estimated frequency for all d attributes
    lst_freq_est = []
    for idx in range(d):
        reports = reports_tuple[:, idx]
        lst_freq_est.append(L_ADP_Aggregator(reports, lst_k[idx], eps_perm_spl, eps_1_spl))

    return np.array(lst_freq_est, dtype='object')
