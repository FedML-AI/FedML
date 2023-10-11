class Client:
    """
    Represents a client in a federated learning system.

    Args:
        client_idx (int): The index of the client.
        local_training_data (list): Local training data.
        local_test_data (list): Local test data.
        local_sample_number (int): Number of local samples.
        args (object): Arguments for configuration.
        device (str): The device (e.g., 'cpu' or 'cuda') for model training.
        model_trainer (object): The model trainer object for training and testing.

    Attributes:
        client_idx (int): The index of the client.
        local_training_data (list): Local training data.
        local_test_data (list): Local test data.
        local_sample_number (int): Number of local samples.
        args (object): Arguments for configuration.
        device (str): The device (e.g., 'cpu' or 'cuda') for model training.
        model_trainer (object): The model trainer object for training and testing.

    Methods:
        update_local_dataset(client_idx, local_training_data, local_test_data, local_sample_number):
            Updates the local dataset for the client.

        get_sample_number(): Gets the number of local samples.

        train(w_global): Trains the client's model using the global model weights.

        local_test(b_use_test_dataset): Tests the client's model using local or test data.

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
        Updates the local dataset for the client.

        Args:
            client_idx (int): The index of the client.
            local_training_data (list): Updated local training data.
            local_test_data (list): Updated local test data.
            local_sample_number (int): Updated number of local samples.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        """
        Gets the number of local samples.

        Returns:
            int: Number of local samples.
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        Trains the client's model using the global model weights.

        Args:
            w_global (object): Global model weights.

        Returns:
            object: Updated client model weights.
        """
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        """
        Tests the client's model using local or test data.

        Args:
            b_use_test_dataset (bool): Flag to use test dataset for testing.

        Returns:
            object: Model evaluation metrics.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
