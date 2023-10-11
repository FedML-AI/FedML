class Client:
    """
    Represents a client in a federated learning setting.

    Args:
        client_idx (int): Index of the client.
        local_training_data: Local training data for the client.
        local_test_data: Local test data for the client.
        local_sample_number: Number of local samples.
        args: Arguments for client configuration.
        device: Device (e.g., 'cuda' or 'cpu') to perform computations.
        model_trainer: Trainer for the client's model.
    """
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

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        """
        Update the local dataset for the client.

        Args:
            client_idx (int): Index of the client.
            local_training_data: New local training data for the client.
            local_test_data: New local test data for the client.
            local_sample_number: New number of local samples.
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
            int: Number of local samples.
        """
        return self.local_sample_number

    def train(self, w_global, grad_global, global_named_states):
        """
        Train the client's model.

        Args:
            w_global: Global model parameters.
            grad_global: Global gradient.
            global_named_states: Named states of the global optimizer.

        Returns:
            tuple: A tuple containing local model weights and local gradients.
        """
        self.model_trainer.set_model_params(w_global)
        local_grad = self.model_trainer.train(self.local_training_data, self.device, self.args, grad_global, global_named_states)
        weights = self.model_trainer.get_model_params()
        return weights, local_grad

    def local_test(self, b_use_test_dataset):
        """
        Perform local testing on the client's dataset.

        Args:
            b_use_test_dataset (bool): Whether to use the test dataset.

        Returns:
            dict: Metrics from the local test.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
