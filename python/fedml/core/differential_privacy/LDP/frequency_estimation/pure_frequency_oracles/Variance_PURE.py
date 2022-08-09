
# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security 17).

def VAR_Pure(p, q, n=1, f=0):

    """
    Variance value of 'pure' frequency oracles from [1]

    :param p: probability of being 'honest';
    :param q: probability of randomizing value/bit;
    :param n: number of users (if known in advance);
    :param f: real frequency of value (if known in advance - not realistic in locally differentially private scenario);
    :return: variance value.
    """

    return q * (1 - q) / (n * (p - q)**2) + f * (1 - p - q) / (n * (p - q))
