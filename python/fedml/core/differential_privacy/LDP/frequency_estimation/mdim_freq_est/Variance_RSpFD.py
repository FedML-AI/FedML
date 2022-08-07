
# [1] Arcolezi et al (2021) "Random Sampling Plus Fake Data: Multidimensional Frequency Estimates With Local Differential Privacy" (ACM CIKM).

def VAR_RSpFD_GRR(p, q, k, d, n=1, f=0):

    """
    Variance value of using RS+FD[GRR] protocol of [1]

    :param p: probability of being 'honest';
    :param q: probability of randomizing value/bit;
    :param k: attribute's domain size;
    :param d: number of attributes;
    :param n: number of users (if known in advance);
    :param f: real frequency of value (if known in advance - not realistic in locally differentially private scenario);
    :return: variance value.
    """

    sig_grr = (1 / d) * (q + f * (p - q) + (d - 1) / k)

    var_grr = ((d**2 * sig_grr * (1 - sig_grr)) / (n * (p - q)**2))

    return var_grr

def VAR_RSpFD_UE_zero(p, q, d, n=1, f=0):

    """
    Variance value of using RS+FD[UE-zero] protocol of [1]

    :param p: probability of being 'honest';
    :param q: probability of randomizing value/bit;
    :param d: number of attributes;
    :param n: number of users (if known in advance);
    :param f: real frequency of value (if known in advance - not realistic in locally differentially private scenario);
    :return: variance value.
    """

    sig_ue = (1/d) * (d*q + f * (p-q))

    var_ue_z = ((d**2 * sig_ue * (1 - sig_ue)) / (n * (p - q)**2))

    return var_ue_z

def VAR_RSpFD_UE_rnd(p, q, k, d, n=1, f=0):

    """
    Variance value of using RS+FD[UE-rnd] protocol of [1]

    :param p: probability of being 'honest';
    :param q: probability of randomizing value/bit;
    :param k: attribute's domain size;
    :param d: number of attributes;
    :param n: number of users (if known in advance);
    :param f: real frequency of value (if known in advance - not realistic in locally differentially private scenario);
    :return: variance value.
    """

    sig_ue = (1 / d) * (f * (p - q) + q + (d - 1) * ((p / k) + ((k - 1)/(k)) * q) )

    var_ue_rnd = ((d**2 * sig_ue * (1 - sig_ue)) / (n * (p - q)**2))

    return var_ue_rnd
