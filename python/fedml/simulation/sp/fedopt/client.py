import logging


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        """
        Initialize a client in the federated learning system.

        Args:
            client_idx (int): The unique identifier for this client.
            local_training_data (torch.Dataset): The local training dataset for this client.
            local_test_data (torch.Dataset): The local test dataset for this client.
            local_sample_number (int): The number of samples in the local training dataset.
            args: Additional arguments and settings.
            device: The device (e.g., CPU or GPU) on which to perform computations.
            model_trainer: The model trainer responsible for training and testing.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        """
        Update the local dataset for this client.

        Args:
            client_idx (int): The unique identifier for this client.
            local_training_data (torch.Dataset): The new local training dataset.
            local_test_data (torch.Dataset): The new local test dataset.
            local_sample_number (int): The number of samples in the new local training dataset.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        """
        Get the number of samples in the local training dataset.

        Returns:
            int: The number of samples in the local training dataset.
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        Train the client's local model.

        Args:
            w_global: The global model weights.

        Returns:
            weights: The updated local model weights.
        """
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        """
        Perform local testing using either the local test dataset or local training dataset.

        Args:
            b_use_test_dataset (bool): If True, use the local test dataset for testing. Otherwise, use the local training dataset.

        Returns:
            metrics: The evaluation metrics obtained during testing.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
