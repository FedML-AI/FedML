class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        """
        Initialize a federated learning client.

        Args:
            client_idx (int): Index of the client.
            local_training_data (dataset): Local training dataset for the client.
            local_test_data (dataset): Local test dataset for the client.
            local_sample_number (int): Number of samples in the local dataset.
            args (argparse.Namespace): Parsed command-line arguments.
            device (torch.device): The device to run training and inference on.
            model_trainer (ModelTrainer): Trainer for the client's machine learning model.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        """
        Update the local dataset and client index.

        Args:
            client_idx (int): New index of the client.
            local_training_data (dataset): New local training dataset for the client.
            local_test_data (dataset): New local test dataset for the client.
            local_sample_number (int): New number of samples in the local dataset.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        """
        Get the number of samples in the local dataset.

        Returns:
            int: Number of samples in the local dataset.
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        Train the client's machine learning model using global model parameters.

        Args:
            w_global (list): Global model parameters.

        Returns:
            list: Updated model parameters after training.
        """
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        """
        Perform local testing on the client's machine learning model.

        Args:
            b_use_test_dataset (bool): Whether to use the test dataset for testing.

        Returns:
            dict: Metrics obtained from local testing.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
