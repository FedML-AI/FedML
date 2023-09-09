from .utils import transform_tensor_to_list


class FedProxTrainer(object):
    """
    Federated Proximal Trainer for model training.

    Args:
        client_index (int): Index of the client.
        train_data_local_dict (dict): Dictionary of local training data.
        train_data_local_num_dict (dict): Dictionary of local training data counts.
        test_data_local_dict (dict): Dictionary of local testing data.
        train_data_num (int): Total number of training data samples.
        device (object): Device for training (e.g., CPU or GPU).
        args (object): Arguments for configuration.
        model_trainer (object): Model trainer for training.

    Attributes:
        trainer (object): Model trainer for training.
        client_index (int): Index of the client.
        train_data_local_dict (dict): Dictionary of local training data.
        train_data_local_num_dict (dict): Dictionary of local training data counts.
        test_data_local_dict (dict): Dictionary of local testing data.
        all_train_data_num (int): Total number of training data samples.
        train_local (object): Local training data for the client.
        local_sample_number (int): Number of local training data samples.
        test_local (object): Local testing data for the client.
        device (object): Device for training.
        args (object): Arguments for configuration.
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
            weights (object): New model weights.
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """
        Update the dataset for training and testing.

        Args:
            client_index (int): Index of the client.
        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        """
        Train the model.

        Args:
            round_idx (int, optional): Index of the training round (default: None).

        Returns:
            tuple: Tuple containing trained model weights and local sample count.
        """
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        """
        Test the trained model.

        Returns:
            tuple: Tuple containing training and testing metrics.
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
