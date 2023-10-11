from .utils import transform_tensor_to_list


class FedOptTrainer(object):
    """Trains a federated optimizer on a client's local data.

    This class is responsible for training a federated optimizer on a client's
    local data. It updates the model using the federated optimization technique
    and returns the updated model weights.

    Attributes:
        trainer: The model trainer used for local training.
        client_index: The index of the client.
        train_data_local_dict: A dictionary containing local training data.
        train_data_local_num_dict: A dictionary containing the number of samples for each client.
        all_train_data_num: The total number of training samples across all clients.
        device: The device (e.g., CPU or GPU) for training.
        args: A configuration object containing training parameters.

    Methods:
        update_model(weights): Updates the model with the provided weights.
        update_dataset(client_index): Updates the dataset for the given client.
        train(round_idx=None): Trains the federated optimizer on the local data.

    """

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
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.all_train_data_num = train_data_num
        

        self.device = device
        self.args = args

    def update_model(self, weights):
        """Update the model with the provided weights.

        Args:
            weights: The updated model weights.

        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """Update the dataset for the given client.

        Args:
            client_index: The index of the client.

        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self, round_idx=None):
        """Train the federated optimizer on the local data.

        Args:
            round_idx: The index of the training round (optional).

        Returns:
            weights: The updated model weights.
            local_sample_number: The number of local training samples.

        """
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number
