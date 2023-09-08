from .utils import transform_tensor_to_list


class FedAVGTrainer(object):
    """
    A class that handles training and testing on a local client in the FedAVG framework.

    This class is responsible for training and testing a local model using client-specific data in a federated learning setting.

    Args:
        client_index: The index or ID of the client.
        train_data_local_dict: A dictionary containing local training data.
        train_data_local_num_dict: A dictionary containing the number of training samples for each client.
        test_data_local_dict: A dictionary containing local testing data.
        train_data_num: The total number of training samples.
        device: The computing device (e.g., "cuda" or "cpu") to perform training and testing.
        args: An object containing configuration parameters.
        model_trainer: A model trainer object responsible for training and testing.

    Attributes:
        trainer: A model trainer object responsible for training and testing.
        client_index: The index or ID of the client.
        train_data_local_dict: A dictionary containing local training data.
        train_data_local_num_dict: A dictionary containing the number of training samples for each client.
        test_data_local_dict: A dictionary containing local testing data.
        all_train_data_num: The total number of training samples.
        train_local: Local training data for the current client.
        local_sample_number: The number of training samples for the current client.
        test_local: Local testing data for the current client.
        device: The computing device (e.g., "cuda" or "cpu") to perform training and testing.
        args: An object containing configuration parameters.

    Methods:
        update_model(weights): Update the model with new weights.
        update_dataset(client_index): Update the local datasets and client index.
        train(round_idx=None): Train the local model using the current client's data.
        test(): Test the local model on both training and testing data.

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
        """Update the model with new weights.

        Args:
            weights: The new model weights to set.
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """Update the local datasets and client index.

        Args:
            client_index: The index or ID of the client.
        """
        self.client_index = client_index

        if self.train_data_local_dict is not None:
            self.train_local = self.train_data_local_dict[client_index]
        else:
            self.train_local = None

        if self.train_data_local_num_dict is not None:
            self.local_sample_number = self.train_data_local_num_dict[client_index]
        else:
            self.local_sample_number = 0

        if self.test_data_local_dict is not None:
            self.test_local = self.test_data_local_dict[client_index]
        else:
            self.test_local = None

    def train(self, round_idx=None):
        """Train the local model using the current client's data.

        Args:
            round_idx: The current communication round index (optional).

        Returns:
            weights: The trained model weights.
            local_sample_number: The number of training samples used for training.
        """
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        """Test the local model on both training and testing data.

        Returns:
            A tuple containing the following metrics:
            - train_tot_correct: The total number of correct predictions on the training data.
            - train_loss: The loss on the training data.
            - train_num_sample: The total number of training samples.
            - test_tot_correct: The total number of correct predictions on the testing data.
            - test_loss: The loss on the testing data.
            - test_num_sample: The total number of testing samples.
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
