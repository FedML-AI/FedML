from .mechanisms import Laplace, Gaussian
import logging


class FedMLDifferentialPrivacy:
    _dp_instance = None

    @staticmethod
    def get_instance():
        if FedMLDifferentialPrivacy._dp_instance is None:
            FedMLDifferentialPrivacy._dp_instance = FedMLDifferentialPrivacy()
        return FedMLDifferentialPrivacy._dp_instance

    def __init__(self):
        self.is_dp_enabled = False
        self.dp_type = None
        self.dp = None

    def init(
        self, args
    ):
        if hasattr(args, "enable_dp") and args.enable_dp:
            logging.info(".......init dp......." + args.mechanism_type + "-" + args.dp_type)
            self.is_dp_enabled = True
            mechanism_type = args.mechanism_type.lower()
            self.dp_type = args.dp_type.lower().strip()
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

    def is_cdp_enabled(self):
        return self.is_enabled() and self.get_dp_type() == "cdp"

    def is_ldp_enabled(self):
        return self.is_enabled() and self.get_dp_type() == "ldp"

    def get_dp_type(self):
        return self.dp_type

    def compute_a_noise(self, size):
        return self.dp.compute_a_noise(size)

    def compute_new_grad(self, grad):
        # print(f"grad = {grad}")
        return self.compute_a_noise(grad.shape) + grad

    # add noise
    # def compute_global_gradient(self, avg_param):
    #     new_grad = dict()
    #     print(f"grad={avg_param}")
    #     for k in avg_param.keys():
    #         new_grad[k] = self.compute_a_noise(avg_param[k].shape)
    #     return new_grad

    def add_cdp_noise(self, avg_param):
        new_grad = dict()
        # print(f"grad={avg_param}")
        for k in avg_param.keys():
            new_grad[k] = self.compute_new_grad(avg_param[k])
        return new_grad

    def add_ldp_noise(self, grads):
        new_grads = []
        for i in range(len(grads)):
            list = []
            for x in grads[i]:
                y = self.compute_new_grad(x)
                list.append(y)
            new_grads.append(tuple(list))
        return new_grads
