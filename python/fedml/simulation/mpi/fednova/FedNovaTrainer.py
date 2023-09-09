from .utils import transform_tensor_to_list


class FedNovaTrainer(object):
    """
    Trainer class for FedNova federated learning.

    Methods:
        __init__: Initialize the FedNovaTrainer.
        update_model: Update the model with global weights.
        update_dataset: Update the local dataset for training.
        get_lr: Calculate the learning rate for the current round.
        train: Train the model on the local dataset.
        test: Evaluate the model on the local training and test datasets.

    Parameters:
        client_index (int): Index of the client.
        train_data_local_dict (dict): Local training dataset for each client.
        train_data_local_num_dict (dict): Number of samples in the local training dataset for each client.
        test_data_local_dict (dict): Local test dataset for each client.
        train_data_num (int): Total number of training samples across all clients.
        device: Device (e.g., GPU or CPU) for model training.
        args: Command-line arguments.
        model_trainer: Trainer for the machine learning model.
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
        """
        Initialize the FedNovaTrainer.

        Args:
            client_index (int): Index of the client.
            train_data_local_dict (dict): Local training dataset for each client.
            train_data_local_num_dict (dict): Number of samples in the local training dataset for each client.
            test_data_local_dict (dict): Local test dataset for each client.
            train_data_num (int): Total number of training samples across all clients.
            device: Device (e.g., GPU or CPU) for model training.
            args: Command-line arguments.
            model_trainer: Trainer for the machine learning model.
        """
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.total_train_num = sum(list(self.train_data_local_num_dict.values()))
        self.device = device
        self.args = args

    def update_model(self, weights):
        """
        Update the model with global weights.

        Args:
            weights: Global model weights.
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """
        Update the local dataset for training.

        Args:
            client_index (int): Index of the client.
        """
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def get_lr(self, progress):
        """
        Calculate the learning rate for the current round.

        Args:
            progress (int): Current round index.

        Returns:
            float: Learning rate.
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
        Train the model on the local dataset.

        Args:
            round_idx (int): Current round index.

        Returns:
            tuple: A tuple containing average loss, normalized gradient, and effective tau.
        """
        self.args.round_idx = round_idx
        avg_loss, norm_grad, tau_eff = self.trainer.train(self.train_local, self.device, self.args,
            ratio=self.local_sample_number / self.total_train_num)
        return avg_loss, norm_grad, tau_eff

    def test(self):
        """
        Evaluate the model on the local training and test datasets.

        Returns:
            tuple: A tuple containing training accuracy, training loss, training sample count,
                   test accuracy, test loss, and test sample count.
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
    