import copy

import torch


def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.concat(param, dim=0)


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

        # self.alpha = args.feddyn_alpha

        self.old_grad = copy.deepcopy(self.model_trainer.get_model_params())
        for key in self.old_grad.keys():
            # self.old_grad[key] = torch.zeros_like(self.old_grad[key]).detach()
            self.old_grad[key] = torch.zeros_like(self.old_grad[key])


    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.old_grad = self.model_trainer.train(self.local_training_data, self.device, self.args, self.old_grad)
        weights = self.model_trainer.get_model_params()
        return weights, self.old_grad

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
