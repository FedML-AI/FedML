from fedml.core.dp.mechanisms import Gaussian, Laplace

"""call dp mechanisms, e.g., Gaussian, Laplace """


class DPMechanism:
    def __init__(self, mechanism_type, epsilon, delta, sensitivity):
        mechanism_type = mechanism_type.lower()
        if mechanism_type == "laplace":
            self.dp = Laplace(
                epsilon=epsilon, delta=delta, sensitivity=sensitivity
            )
        elif mechanism_type == "gaussian":
            self.dp = Gaussian(epsilon, delta=delta, sensitivity=sensitivity)
        else:
            raise NotImplementedError("DP mechanism not implemented!")

    def add_noise(self, grad):
        new_grad = dict()
        for k in grad.keys():
            new_grad[k] = self._compute_new_grad(grad[k])
        # if self.enable_accountant:
        #     self.accountant.spend(epsilon=self.epsilon, delta=0)
        return new_grad

    def _compute_new_grad(self, grad):
        noise = self.dp.compute_noise(grad.shape)
        return noise + grad

    # def add_noise(self, grad):
    #     noise_list_len = len(vectorize_weight(grad))
    #     noise_list = np.zeros(noise_list_len)
    #     vec_weight = vectorize_weight(grad)
    #     for i in range(noise_list_len):
    #         noise_list[i] = self.dp.compute_noise()
    #     new_vec_grad = vec_weight + noise_list
    #
    #     new_grad = {}
    #     index_bias = 0
    #     print(f"noises in add_noise = {noise_list}")
    #     for item_index, (k, v) in enumerate(grad.items()):
    #         if is_weight_param(k):
    #             new_grad[k] = new_vec_grad[index_bias : index_bias + v.numel()].view(
    #                 v.size()
    #             )
    #             index_bias += v.numel()
    #         else:
    #             new_grad[k] = v
    #     return new_grad

    def add_a_noise_to_local_data(self, local_data):
        new_data = []
        for i in range(len(local_data)):
            list = []
            for x in local_data[i]:
                y = self._compute_new_grad(x)
                list.append(y)
            new_data.append(tuple(list))
        return new_data