from .utils import transform_tensor_to_list


class AsyncFedAVGTrainer(object):
    """
    An asynchronous Federated Averaging trainer for client nodes in a federated learning system.

    Args:
        client_index (int): The index of the client node.
        train_data_local_dict (dict): A dictionary containing local training data for each client.
        train_data_local_num_dict (dict): A dictionary containing the number of local training samples for each client.
        test_data_local_dict (dict): A dictionary containing local testing data for each client.
        train_data_num (int): The total number of training samples across all clients.
        device (torch.device): The device (e.g., CPU or GPU) to perform training and testing on.
        args (argparse.Namespace): Command-line arguments and configurations for training.
        model_trainer (ClientTrainer): An instance of a client-side model trainer.

    Attributes:
        trainer (ClientTrainer): The model trainer used for training and testing.
        client_index (int): The index of the client node.
        train_data_local_dict (dict): A dictionary containing local training data for each client.
        train_data_local_num_dict (dict): A dictionary containing the number of local training samples for each client.
        test_data_local_dict (dict): A dictionary containing local testing data for each client.
        all_train_data_num (int): The total number of training samples across all clients.
        train_local (Dataset): The local training dataset for the current client.
        local_sample_number (int): The number of local training samples for the current client.
        test_local (Dataset): The local testing dataset for the current client.
        device (torch.device): The device used for training and testing.
        args (argparse.Namespace): Command-line arguments and configurations for training.

    Methods:
        update_model(weights):
            Update the model's weights with the provided weights.

        update_dataset(client_index):
            Update the local training and testing datasets for the current client.

        train(round_idx=None):
            Train the model on the local training dataset.

        test():
            Test the model on both the local training and testing datasets.

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
        Update the model's weights with the provided weights.

        Args:
            weights (dict): The model parameters as a dictionary of tensors.

        Returns:
            None
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """
        Update the local training and testing datasets for the current client.

        Args:
            client_index (int): The index of the current client.

        Returns:
            None
        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        """
        Train the model on the local training dataset.

        Args:
            round_idx (int, optional): The current round index. Defaults to None.

        Returns:
            tuple: A tuple containing the trained model's weights and the number of local training samples.
        """
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        """
        Test the model on both the local training and testing datasets.

        Returns:
            tuple: A tuple containing various metrics, including training accuracy, training loss, the number
            of training samples, testing accuracy, testing loss, and the number of testing samples.
        """
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
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
