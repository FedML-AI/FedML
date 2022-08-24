from .mechanisms import Laplace, Gaussian


class FedMLDifferentialPrivacy:
    _defender_instance = None

    @staticmethod
    def get_instance():
        if FedMLDifferentialPrivacy._defender_instance is None:
            FedMLDifferentialPrivacy._defender_instance = FedMLDifferentialPrivacy()
        return FedMLDifferentialPrivacy._defender_instance

    def __init__(self):
        self.is_dp_enabled = False
        self.dp_type = None
        self.dp = None

    def init(
        self, args
    ):
        if hasattr(args, "enable_dp") and args.enable_dp:
            self.is_dp_enabled = True
            mechanism_type = args.mechanism_type.lower()
            self.dp_type = args.dp_type.lower()
            if self.dp_type not in ["cdp", "ldp"]:
                raise ValueError("DP type can only be cdp (for central DP) and ldp (for local DP)! ")
            if mechanism_type == "laplace":
                self.dp = Laplace(epsilon=args.epsilon, delta=args.delta, sensitivity=args.sensitivity)
            elif mechanism_type == "gaussian":
                self.dp = Gaussian(epsilon=args.epsilon, delta=args.delta, sensitivity=args.sensitivity)
            else:
                raise NotImplementedError("DP mechanism not implemented!")

    def is_enabled(self):
        return self.is_dp_enabled

    def get_dp_type(self):
        return self.dp_type

    def compute_a_noise(self, size):
        return self.dp.compute_a_noise(size)

    # add noise
    def compute_randomized_gradient(self, grad):
        new_grad = dict()
        for k in grad.keys():
            new_grad[k] = self.compute_a_noise(grad[k].shape) + grad[k]
        return new_grad

    def add_cdp_noise(self, avg_param):
        if self.dp_type == "cdp":
            avg_param = self.compute_randomized_gradient(avg_param)
        return avg_param

    def add_ldp_noise(self, grad):
        if self.dp_type == "ldp":
            grad = self.compute_randomized_gradient(grad)
        return grad
