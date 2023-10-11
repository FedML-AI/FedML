class Client:
    """
    Represents a federated learning client.

    Args:
        client_idx (int): Index of the client.
        local_training_data (Dataset): Local training dataset for the client.
        local_test_data (Dataset): Local test dataset for the client.
        local_sample_number (int): Number of local training samples.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device for training (e.g., "cpu" or "cuda").
        model_trainer (ModelTrainer): Trainer for the client's model.
    """

    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer
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
            local_training_data (Dataset): New local training dataset.
            local_test_data (Dataset): New local test dataset.
            local_sample_number (int): Number of local training samples.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        """
        Get the number of local training samples.

        Returns:
            int: Number of local training samples.
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        Train the client's model using the global model weights.

        Args:
            w_global (dict): Global model weights.

        Returns:
            dict: Updated client model weights.
        """
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        """
        Test the client's model on the local test dataset.

        Args:
            b_use_test_dataset (bool): Flag to indicate whether to use the test dataset.

        Returns:
            dict: Evaluation metrics.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
