from .utils import transform_tensor_to_list


class FedOptTrainer(object):
    """Trains a federated learning model for a specific client."""

    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    ):
        """Initialize the FedOptTrainer.

        Args:
            client_index (int): The index of the client.
            train_data_local_dict (dict): A dictionary mapping client indexes to their local training datasets.
            train_data_local_num_dict (dict): A dictionary mapping client indexes to the number of samples in their local datasets.
            train_data_num (int): The total number of training samples.
            device (str): The device (e.g., 'cuda' or 'cpu') on which to perform training.
            args (object): Configuration parameters for training.
            model_trainer (object): An instance of the model trainer for this client.
        """
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num

        self.device = device
        self.args = args

    def update_model(self, weights):
        """Update the model parameters.

        Args:
            weights (dict): A dictionary containing the updated model parameters.
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """Update the local dataset for the client.

        Args:
            client_index (int): The index of the client whose dataset should be updated.
        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self, round_idx=None):
        """Train the federated learning model for the client.

        Args:
            round_idx (int, optional): The current federated learning round index. Defaults to None.

        Returns:
            tuple: A tuple containing the updated model weights and the number of local samples used for training.
        """
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number
