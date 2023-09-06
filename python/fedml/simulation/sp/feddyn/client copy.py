import copy

import torch


def model_parameter_vector(model):
    """
    Flatten the model's parameters into a single vector.

    Args:
        model (torch.nn.Module): The neural network model.

    Returns:
        torch.Tensor: A flattened vector containing all model parameters.
    """
    param = [p.view(-1) for p in model.parameters()]
    return torch.concat(param, dim=0)


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        """
        Initialize a client for federated learning.

        Args:
            client_idx (int): Index of the client.
            local_training_data (torch.utils.data.DataLoader): Local training data.
            local_test_data (torch.utils.data.DataLoader): Local test data.
            local_sample_number (int): Number of samples in the local dataset.
            args: Command-line arguments.
            device (torch.device): Device for training.
            model_trainer: Model trainer for training and testing.
        """
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
        """
        Update the local dataset for the client.

        Args:
            client_idx (int): Index of the client.
            local_training_data (torch.utils.data.DataLoader): Local training data.
            local_test_data (torch.utils.data.DataLoader): Local test data.
            local_sample_number (int): Number of samples in the local dataset.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        """
        Get the number of samples in the local dataset.

        Returns:
            int: Number of samples.
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        Train the client's model using the global model parameters.

        Args:
            w_global: Global model parameters.

        Returns:
            tuple: A tuple containing the updated weights and gradients.
        """
        self.model_trainer.set_model_params(w_global)
        self.old_grad = self.model_trainer.train(self.local_training_data, self.device, self.args, self.old_grad)
        weights = self.model_trainer.get_model_params()
        return weights, self.old_grad

    def local_test(self, b_use_test_dataset):
        """
        Perform local testing on the client's dataset.

        Args:
            b_use_test_dataset (bool): Whether to use the test dataset or training dataset.

        Returns:
            dict: Metrics from the local test.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
