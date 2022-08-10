from .mechanisms import Laplace, Gaussian


class DP:
    def __init__(
        self, epsilon, delta=0, sensitivity=1.0, mechanism_type="laplace"
    ):
        mechanism_type = mechanism_type.lower()
        if mechanism_type == "laplace":
            self.dp = Laplace(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        elif mechanism_type == "gaussian":
            self.dp = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

    def randomise(self, value):
        return self.dp.randomise(value)

    def compute_a_noise(self):
        return self.dp.compute_a_noise()

    # add noise
    def compute_randomized_gradient(self, grad):
        new_grad = dict()
        for k in grad.keys():
            new_grad[k] = self.compute_a_noise() + grad[k]
        return new_grad