import numpy as np
from sys import maxsize
import xxhash

# [1] Wang et al (2017) "Locally differentially private protocols for frequency estimation" (USENIX Security).
# [2] Bassily and Smith "Local, private, efficient protocols for succinct histograms" (STOC).

# Code adapted from pure-ldp library (https://github.com/Samuel-Maddock/pure-LDP) developed by Samuel Maddock


class LH:
    def __init__(self, attr_domain_size, epsilon, optimal=True):
        if epsilon is None or attr_domain_size is None:
            raise ValueError(
                "attr_domain_size (int) and epsilon (float) need a numerical value."
            )
        self.epsilon = epsilon
        self.attr_domain_size = attr_domain_size

        # Binary LH (BLH) parameter
        self.g = 2
        # Optimal LH (OLH) parameter
        if optimal:
            self.g = int(round(np.exp(epsilon))) + 1
        # LH parameters with reduced domain size g
        self.p = np.exp(epsilon) / (np.exp(epsilon) + self.g - 1)
        self.q = 1 / (np.exp(epsilon) + self.g - 1)

    def client_permute(self, input_data):

        """
        Local Hashing (LH) protocol[1], which is logically equivalent to the random matrix projection technique in [2].
        """

        # Generate random seed and hash the user's value
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
        hashed_input_data = (
            xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % self.g
        )

        # LH perturbation function (i.e., GRR-based)
        sanitized_value = hashed_input_data
        rnd = np.random.random()
        if rnd > self.p - self.q:

            sanitized_value = np.random.randint(0, self.g)

        return (sanitized_value, rnd_seed)

    def server_aggregate(self, reports):
        if len(reports) == 0:
            raise ValueError("List of reports is empty.")

        # Number of reports
        n = len(reports)

        # Count how many times each value has been reported
        count_report = np.zeros(self.attr_domain_size)
        for tuple_val_seed in reports:
            for v in range(self.attr_domain_size):
                if tuple_val_seed[0] == (
                    xxhash.xxh32(str(v), seed=tuple_val_seed[1]).intdigest() % self.g
                ):
                    count_report[v] += 1

        # Ensure non-negativity of estimated frequency
        a = self.g / (self.p * self.g - 1)
        b = n / (self.p * self.g - 1)
        est_freq = (a * count_report - b).clip(0)

        # Re-normalized estimated frequency
        norm_est_freq = np.nan_to_num(est_freq / sum(est_freq))

        return norm_est_freq
