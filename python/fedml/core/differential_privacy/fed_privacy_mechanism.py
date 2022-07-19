from fedml.core.differential_privacy.mechanisms import Laplace, Gaussian, Binary, Bingham, Exponential, Geometric, \
    Staircase


class FedPrivacyMechanism:
    def __init__(self, epsilon, delta=0, sensitivity=1.0, type="CDP", mechanism_type="laplace"):  #, binary_label0="0", binary_label1="1", utility, candidates=None, measure=None):
        self.dp_type = type
        mechanism_type = mechanism_type.lower()
        if mechanism_type == "laplace":
            self.dp = Laplace(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        elif mechanism_type == "gaussian":
            self.dp = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        # elif mechanism_type == "binary":
        #     self.dp = Binary(epsilon=epsilon, value0=binary_label0, value1=binary_label1)
        # elif mechanism_type == "bingham":
        #     self.dp = Bingham(epsilon=epsilon, sensitivity=1.0)
        # elif mechanism_type == "exponential":
        #     self.dp = Exponential(epsilon=epsilon, sensitivity=sensitivity, utility=utility, candidates=candidates, measure=measure)
        # elif mechanism_type == "geometric":
        #     self.dp = Geometric(epsilon=epsilon, sensitivity=sensitivity)
        # elif mechanism_type == "":
        #     self.dp = Staircase(epsilon=epsilon, sensitivity=sensitivity, gamma=gamma)

    def randomise(self, value):
        return self.dp.randomise(value)

    # add noise
    def compute_randomized_weights(self, weights):
        return weights + self.randomise(0)