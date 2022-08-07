
# [1] Arcolezi et al (2021) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (arXiv:2111.04636).
# [2] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).

def VAR_Long_Pure(p1, q1, p2, q2, n=1, f=0):

    """
    Variance value of longitudinal 'pure' frequency oracles from [1] following the memoization-based framework from [2]

    :param p1: probability of being 'honest' for first round of sanitization (permanent memoization);
    :param q1: probability of randomizing value/bit for round of sanitization (permanent memoization);
    :param p2: probability of being 'honest' for second round of sanitization;
    :param q2: probability of randomizing value/bit for second round of sanitization;
    :param n: number of users (if known in advance);
    :param f: real frequency of value (if known in advance - not realistic in locally differentially private scenario);
    :return: variance value.
    """
    
    sig = f * (2 * p1 * p2 - 2 * p1 * q2 + 2 * q2 - 1) + p2 * q1 + q2 * (1 - q1)

    return sig * (1 - sig) / (n * (p1 - q1)**2 * (p2 - q2)**2)
