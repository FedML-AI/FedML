import copy

import torch


def model_parameter_vector(model):
    """
    Flatten and concatenate the model parameters into a single vector.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        torch.Tensor: The concatenated parameter vector.
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
            client_idx (int): The index of the client.
            local_training_data (torch.utils.data.DataLoader): The local training dataset.
            local_test_data (torch.utils.data.DataLoader): The local test dataset.
            local_sample_number (int): The number of local samples.
            args: The command-line arguments.
            device (torch.device): The device (e.g., "cuda" or "cpu") for computation.
            model_trainer: The model trainer responsible for training and testing.
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
        Update the client's local dataset.

        Args:
            client_idx (int): The index of the client.
            local_training_data (torch.utils.data.DataLoader): The new local training dataset.
            local_test_data (torch.utils.data.DataLoader): The new local test dataset.
            local_sample_number (int): The number of local samples in the new dataset.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        """
        Get the number of local samples.

        Returns:
            int: The number of local samples.
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        Train the client's model using global weights.

        Args:
            w_global: The global model weights.

        Returns:
            dict: The updated client's model weights.
        """
        self.model_trainer.set_model_params(w_global)
        self.old_grad = self.model_trainer.train(self.local_training_data, self.device, self.args, self.old_grad)
        weights = self.model_trainer.get_model_params()
        # return weights, self.old_grad
        return weights

    def local_test(self, b_use_test_dataset):
        """
        Perform local testing on the client's model.

        Args:
            b_use_test_dataset (bool): Whether to use the test dataset for testing.

        Returns:
            dict: Test metrics including correctness, loss, and more.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
