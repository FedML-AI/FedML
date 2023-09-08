from .utils import transform_tensor_to_list


class FedGANTrainer(object):
    """
    Trainer for a federated GAN client.

    Args:
        client_index (int): Index of the client.
        train_data_local_dict (dict): Dictionary of local training datasets.
        train_data_local_num_dict (dict): Dictionary of local training dataset sizes.
        test_data_local_dict (dict): Dictionary of local test datasets.
        train_data_num (int): Number of samples in the global training dataset.
        device: Device for training (e.g., 'cuda' or 'cpu').
        args: Configuration arguments.
        model_trainer: Trainer for the GAN model.

    Attributes:
        trainer: Trainer for the GAN model.
        client_index (int): Index of the client.
        train_data_local_dict (dict): Dictionary of local training datasets.
        train_data_local_num_dict (dict): Dictionary of local training dataset sizes.
        test_data_local_dict (dict): Dictionary of local test datasets.
        all_train_data_num (int): Number of samples in the global training dataset.
        train_local: Local training dataset.
        local_sample_number: Number of samples in the local training dataset.
        test_local: Local test dataset.
        device: Device for training (e.g., 'cuda' or 'cpu').
        args: Configuration arguments.
    """

    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None
        self.device = device
        self.args = args

    def update_model(self, weights):
        """
        Update the model with new weights.

        Args:
            weights: New model weights.
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """
        Update the client's dataset.

        Args:
            client_index (int): Index of the client.
        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self, round_idx=None):
        """
        Train the client's GAN model.

        Args:
            round_idx: Index of the training round (optional).

        Returns:
            weights: Updated model weights.
            local_sample_number: Number of samples in the local dataset.
        """
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)
        weights = self.trainer.get_model_params()
        return weights, self.local_sample_number

    def test(self):
        """
        Test the client's GAN model on both training and test datasets.

        Returns:
            Tuple containing:
                - train_tot_correct: Total correct predictions on the training dataset.
                - train_loss: Loss on the training dataset.
                - train_num_sample: Number of samples in the training dataset.
                - test_tot_correct: Total correct predictions on the test dataset.
                - test_loss: Loss on the test dataset.
                - test_num_sample: Number of samples in the test dataset.
        """
        # Train data metrics
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # Test data metrics
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )

