import numpy as np

# [1] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [2] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).


class UE:
    def __init__(self, attr_domain_size, epsilon, optimal=True):
        # domain size: # of distinct values of the attribute
        # param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
        if epsilon is None or attr_domain_size is None:
            raise ValueError("attr_domain_size (int) and epsilon (float) need a numerical value.")
        self.epsilon = epsilon
        self.attr_domain_size = attr_domain_size
        # Symmetric parameters (p+q = 1)
        self.p = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
        self.q = 1 - self.p

        # Optimized parameters
        if optimal:
            self.p = 1 / 2
            self.q = 1 / (np.exp(epsilon) + 1)

    def client_permute(self, input_data):
        """
        Unary Encoding (UE) protocol, a.k.a. Basic One-Time RAPPOR (if optimal=False) [1]

        :param input_data: user's true value;
        :return: sanitized UE vector.
        """

        # Unary encoding
        input_ue_data = np.zeros(self.attr_domain_size)
        if input_data != None:
            input_ue_data[input_data] = 1

        # Initializing a zero-vector
        sanitized_vec = np.zeros(self.attr_domain_size)

        # UE perturbation function
        for ind in range(self.attr_domain_size):
            if input_ue_data[ind] != 1:
                rnd = np.random.random()
                if rnd <= self.q:
                    sanitized_vec[ind] = 1
            else:
                rnd = np.random.random()
                if rnd <= self.p:
                    sanitized_vec[ind] = 1
        return sanitized_vec

    def server_aggregate(self, reports):

        """
        Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

        :param reports: list of all UE-based sanitized vectors;
        :param epsilon: privacy guarantee;
        :param optimal: if True, it uses the Optimized UE (OUE) protocol from [2];
        :return: normalized frequency (histogram) estimation.
        """

        if len(reports) == 0:

            raise ValueError("List of reports is empty.")

        else:
            # Number of reports
            n = len(reports)

            # Ensure non-negativity of estimated frequency
            est_freq = np.array((sum(reports) - self.q * n) / (self.p - self.q)).clip(0)

            # Re-normalized estimated frequency
            norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

            return norm_est_freq
