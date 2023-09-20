import time

from fedml.data import split_data_for_dist_trainers
from ...constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from ...core.mlops.mlops_profiler_event import MLOpsProfilerEvent


class FedMLTrainer(object):
    """
    A class representing a Federated Machine Learning Trainer.

    This class manages the training process for federated learning on a client.

    Args:
        client_index: The index of the client.
        train_data_local_dict: A dictionary mapping client IDs to their local training data.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
        test_data_local_dict: A dictionary mapping client IDs to their local testing data.
        train_data_num: The total number of training data samples.
        device: The device for computations.
        args: The command-line arguments.
        model_trainer: The model trainer.

    Attributes:
        trainer: The model trainer.
        client_index: The index of the client.
        train_data_local_dict: A dictionary mapping client IDs to their local training data.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
        test_data_local_dict: A dictionary mapping client IDs to their local testing data.
        all_train_data_num: The total number of training data samples.
        train_local: The local training data for the client.
        local_sample_number: The number of local training data samples.
        test_local: The local testing data for the client.
        device: The device for computations.
        args: The command-line arguments.

    Methods:
        update_model: Update the federated learning model with new weights.
        update_dataset: Update the local dataset for training.
        train: Train the federated learning model for a specified round.
        test: Test the federated learning model.
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
        Initialize a Federated Machine Learning Trainer.

        Args:
            client_index: The index of the client.
            train_data_local_dict: A dictionary mapping client IDs to their local training data.
            train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
            test_data_local_dict: A dictionary mapping client IDs to their local testing data.
            train_data_num: The total number of training data samples.
            device: The device for computations.
            args: The command-line arguments.
            model_trainer: The model trainer.
        """
        self.trainer = model_trainer

        self.client_index = client_index

        if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            self.train_data_local_dict = split_data_for_dist_trainers(
                train_data_local_dict, args.n_proc_in_silo)
        else:
            self.train_data_local_dict = train_data_local_dict

        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args
        self.args.device = device

    def update_model(self, weights):
        """
        Update the federated learning model with new weights.

        Args:
            weights: The new model weights.
        """
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        """
        Update the local dataset for training.

        Args:
            client_index: The index of the client.
        """
        self.client_index = client_index

        if self.train_data_local_dict is not None:
            if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
                self.train_local = self.train_data_local_dict[client_index][self.args.proc_rank_in_silo]
            else:
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

        self.trainer.update_dataset(
            self.train_local, self.test_local, self.local_sample_number)

    def train(self, round_idx=None):
        """
        Train the federated learning model for a specified round.

        Args:
            round_idx: The index of the training round (optional).

        Returns:
            tuple: A tuple containing weights and the number of local training data samples.
        """
        self.args.round_idx = round_idx
        tick = time.time()

        self.trainer.on_before_local_training(
            self.train_local, self.device, self.args)
        self.trainer.train(self.train_local, self.device, self.args)
        self.trainer.on_after_local_training(
            self.train_local, self.device, self.args)

        MLOpsProfilerEvent.log_to_wandb(
            {"Train/Time": time.time() - tick, "round": round_idx})
        weights = self.trainer.get_model_params()
        # transform Tensor to list
        return weights, self.local_sample_number

    def test(self):
        """
        Test the federated learning model.

        Returns:
            tuple: A tuple containing training accuracy, training loss, the number of training samples,
                   testing accuracy, testing loss, and the number of testing samples.
        """
        # train data
        train_metrics = self.trainer.test(
            self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(
            self.test_local, self.device, self.args)
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
