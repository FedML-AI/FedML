from .utils import transform_tensor_to_list


class FedAVGTrainer(object):
    """
    Trainer class for federated learning clients using the FedAVG algorithm.

    Args:
        client_index (int): The index of the client.
        train_data_local_dict (dict): A dictionary containing local training datasets.
        train_data_local_num_dict (dict): A dictionary containing the number of samples for each local dataset.
        test_data_local_dict (dict): A dictionary containing local testing datasets.
        train_data_num (int): The total number of training samples.
        device (str): The device (e.g., "cpu" or "cuda") for training.
        args (Namespace): Command-line arguments and configuration.
        model_trainer (object): An instance of the model trainer used for training.

    Attributes:
        trainer (object): The model trainer instance.
        client_index (int): The index of the client.
        train_data_local_dict (dict): A dictionary containing local training datasets.
        train_data_local_num_dict (dict): A dictionary containing the number of samples for each local dataset.
        test_data_local_dict (dict): A dictionary containing local testing datasets.
        all_train_data_num (int): The total number of training samples.
        train_local (Dataset): The local training dataset.
        local_sample_number (int): The number of local training samples.
        test_local (Dataset): The local testing dataset.
        device (str): The device for training (e.g., "cpu" or "cuda").
        args (Namespace): Command-line arguments and configuration.

    Methods:
        update_model(weights):
            Update the model with given weights.

        update_dataset(client_index):
            Update the current dataset for training and testing.

        get_lr(progress):
            Calculate the learning rate based on the training progress.

        train(round_idx=None):
            Train the model on the local dataset for a given round.

        test():
            Evaluate the trained model on both local training and testing datasets.

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
        Update the model with the provided weights.

        Args:
            weights (dict): The model parameters to set.

        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """
        Update the current dataset for training and testing.

        Args:
            client_index (int): The index of the client representing the dataset to be used.

        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def get_lr(self, progress):
        """
        Calculate the learning rate based on the training progress.

        Args:
            progress (int): The training progress, typically the round index.

        Returns:
            float: The calculated learning rate.

        """
        # This aims to make a float step_size work.
        if self.args.lr_schedule == "StepLR":
            exp_num = progress / self.args.lr_step_size
            lr = self.args.learning_rate * (self.args.lr_decay_rate**exp_num)
        elif self.args.lr_schedule == "MultiStepLR":
            index = 0
            for milestone in self.args.lr_milestones:
                if progress < milestone:
                    break
                else:
                    index += 1
            lr = self.args.learning_rate * (self.args.lr_decay_rate**index)
        elif self.args.lr_schedule == "None":
            lr = self.args.learning_rate
        else:
            raise NotImplementedError
        return lr

    def train(self, round_idx=None):
        """
        Train the model on the local dataset for a given round.

        Args:
            round_idx (int, optional): The current round index. Defaults to None.

        Returns:
            tuple: A tuple containing the trained model weights and the number of local samples used.

        """
        self.args.round_idx = round_idx
        # lr = self.get_lr(round_idx)
        # self.trainer.train(self.train_local, self.device, self.args, lr=lr)
        self.trainer.train(self.train_local, self.device, self.args)
        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        """
        Evaluate the trained model on both local training and testing datasets.

        Returns:
            tuple: A tuple containing training and testing metrics, including correct predictions, loss, and sample counts.

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
