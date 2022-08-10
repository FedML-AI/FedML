import numpy as np

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Kairouz, Bonawitz, and Ramage (2016) "Discrete distribution estimation under local privacy" (ICML)


class GRR:
    def __init__(self, attr_domain_size, epsilon):
        if epsilon is None or attr_domain_size is None:
            raise ValueError("attr_domain_size (int) and epsilon (float) need a numerical value.")
        self.attr_domain_size = attr_domain_size
        self.epsilon = epsilon
        # GRR parameters
        self.p = np.exp(self.epsilon) / (
            np.exp(self.epsilon) + self.attr_domain_size - 1
        )
        self.q = (1 - self.p) / (self.attr_domain_size - 1)

    def client_permute(self, input_data):
        """
        Generalized Randomized Response (GRR) protocol, i.e., direct encoding [1] or k-RR [2].

        :param input_data: user's true value;
        :return: sanitized value.
        """
        rnd = np.random.random()
        if rnd <= self.p:
            return input_data
        else:
            # Mapping domain size to the range [0, ..., attr_domain_size-1]
            domain = np.arange(self.attr_domain_size)
            return np.random.choice(domain[domain != input_data])

    def server_aggregate(self, reports):
        """
        Statistical Estimator for Normalized Frequency (0 -- 1) with post-processing to ensure non-negativity.

        :param reports: list of all GRR-based sanitized values;
        :return: normalized frequency (histogram) estimation.
        """

        if len(reports) == 0:
            raise ValueError("List of reports is empty.")

        # Number of reports
        n = len(reports)

        # Count how many times each value has been reported
        count_report = np.zeros(self.attr_domain_size)
        for rep in reports:
            count_report[rep] += 1

        # Ensure non-negativity of estimated frequency
        est_freq = np.array((count_report - n * self.q) / (self.p - self.q)).clip(0)

        # Re-normalized estimated frequency
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))
        return norm_est_freq
