from .mechanisms import Laplace, Gaussian


class FedPrivacyMechanism:
    def __init__(
        self, epsilon, delta=0, sensitivity=1.0, type="CDP", mechanism_type="laplace"
    ):
        self.dp_type = type
        mechanism_type = mechanism_type.lower()
        if mechanism_type == "laplace":
            self.dp = Laplace(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        elif mechanism_type == "gaussian":
            self.dp = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

    def randomise(self, value):
        return self.dp.randomise(value)

    # add noise
    def compute_randomized_weights(self, weights):
        return self.randomise(weights)
