from .utils import transform_tensor_to_list


class FedSegTrainer(object):
    """
    Trainer for federated segmentation models on a client.

    This class manages the training process of a federated segmentation model on a client.

    Args:
        client_index (int): The index of the client within the federated system.
        train_data_local_dict (dict): A dictionary containing local training data for each client.
        train_data_local_num_dict (dict): A dictionary containing the number of local training samples for each client.
        train_data_num (int): Total number of training samples across all clients.
        test_data_local_dict (dict): A dictionary containing local test data for each client.
        device (torch.device): The device on which to perform training and evaluation.
        model (nn.Module): The segmentation model to be trained.
        args: Additional configuration arguments.
        model_trainer: Trainer for the segmentation model.

    Attributes:
        args: Additional configuration arguments.
        trainer: Trainer for the segmentation model.
        client_index (int): The index of the client.
        train_data_local_dict (dict): A dictionary containing local training data for the client.
        train_data_local_num_dict (dict): A dictionary containing the number of local training samples for each client.
        test_data_local_dict (dict): A dictionary containing local test data for the client.
        all_train_data_num (int): Total number of training samples across all clients.
        train_local: Local training data for the client.
        local_sample_number (int): The number of local training samples for the client.
        test_local: Local test data for the client.

    Methods:
        update_model(weights): Update the model with the provided weights.
        update_dataset(client_index): Update the dataset for the client with the given index.
        train(): Perform training on the local dataset and return trained weights and the number of local samples.
        test(): Perform testing on the local test dataset and return evaluation metrics.
    """
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        test_data_local_dict,
        device,
        model,
        args,
        model_trainer,
    ):
        """
        Initialize the FedSegTrainer for a client.

        Args:
            client_index (int): The index of the client.
            train_data_local_dict (dict): A dictionary containing local training data for each client.
            train_data_local_num_dict (dict): A dictionary containing the number of local training samples for each client.
            train_data_num (int): Total number of training samples across all clients.
            test_data_local_dict (dict): A dictionary containing local test data for each client.
            device (torch.device): The device on which to perform training and evaluation.
            model: The segmentation model to be trained.
            args: Additional configuration arguments.
            model_trainer: Trainer for the segmentation model.
        """
        self.args = args
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]
        self.args.round_idx = 0

        self.device = device

    def update_model(self, weights):
        """
        Update the model with the provided weights.

        Args:
            weights: Model weights to be set.

        Notes:
            This function updates the model with the provided weights.
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """
        Update the dataset for the client with the given index.

        Args:
            client_index (int): The index of the client.

        Notes:
            This function updates the dataset and client-related attributes for the specified client index.
        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self):
        """
        Perform training on the local dataset and return trained weights and the number of local samples.

        Returns:
            tuple: A tuple containing trained model weights and the number of local training samples.
        """

        self.trainer.train(self.train_local, self.device)
        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        """
        Perform testing on the local test dataset and return evaluation metrics.

        Returns:
            tuple: A tuple containing evaluation metrics on the local test dataset.
        """
        train_evaluation_metrics = None

        if self.args.round_idx and self.args.round_idx % self.args.evaluation_frequency == 0:
            train_evaluation_metrics = self.trainer.test(self.train_local, self.device)

        test_evaluation_metrics = self.trainer.test(self.test_local, self.device)
        self.args.round_idx += 1
        return train_evaluation_metrics, test_evaluation_metrics
