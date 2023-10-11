import logging

from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from .fedml_trainer import FedMLTrainer
from ...ml.trainer.trainer_creator import create_model_trainer
from ...ml.engine import ml_engine_adapter


class TrainerDistAdapter:
    """
    A class representing a Trainer Distribution Adapter for federated learning.

    This adapter facilitates training a federated learning model with distributed computing support.

    Args:
        args: The command-line arguments.
        device: The device for computations.
        client_rank: The rank of the client.
        model: The federated learning model.
        train_data_num: The total number of training data samples.
        train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
        train_data_local_dict: A dictionary mapping client IDs to their local training data.
        test_data_local_dict: A dictionary mapping client IDs to their local testing data.
        model_trainer: The model trainer (optional).

    Attributes:
        process_group_manager: The process group manager for distributed training.
        client_index: The index of the client.
        client_rank: The rank of the client.
        device: The device for computations.
        trainer: The federated learning trainer.
        args: The command-line arguments.

    Methods:
        get_trainer: Get the federated learning trainer.
        train: Train the federated learning model for a round.
        update_model: Update the federated learning model with new parameters.
        update_dataset: Update the dataset for training.
        cleanup_pg: Clean up the process group for distributed training.
    """

    def __init__(
        self,
        args,
        device,
        client_rank,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        """
        Initialize a Trainer Distribution Adapter.

        Args:
            args: The command-line arguments.
            device: The device for computations.
            client_rank: The rank of the client.
            model: The federated learning model.
            train_data_num: The total number of training data samples.
            train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
            train_data_local_dict: A dictionary mapping client IDs to their local training data.
            test_data_local_dict: A dictionary mapping client IDs to their local testing data.
            model_trainer: The model trainer (optional).
        """

        ml_engine_adapter.model_to_device(args, model, device)

        if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            self.process_group_manager, model = ml_engine_adapter.model_ddp(
                args, model, device)

        if model_trainer is None:
            model_trainer = create_model_trainer(model, args)
        else:
            model_trainer.model = model

        client_index = client_rank - 1

        model_trainer.set_id(client_index)

        logging.info("Initiating Trainer")
        trainer = self.get_trainer(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            test_data_local_dict,
            train_data_num,
            device,
            args,
            model_trainer,
        )
        self.client_index = client_index
        self.client_rank = client_rank
        self.device = device
        self.trainer = trainer
        self.args = args

    def get_trainer(
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
        Get the federated learning trainer.

        Args:
            client_index: The index of the client.
            train_data_local_dict: A dictionary mapping client IDs to their local training data.
            train_data_local_num_dict: A dictionary mapping client IDs to the number of local training data samples.
            test_data_local_dict: A dictionary mapping client IDs to their local testing data.
            train_data_num: The total number of training data samples.
            device: The device for computations.
            args: The command-line arguments.
            model_trainer: The model trainer.

        Returns:
            FedMLTrainer: The federated learning trainer.
        """
        return FedMLTrainer(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            test_data_local_dict,
            train_data_num,
            device,
            args,
            model_trainer,
        )

    def train(self, round_idx):
        """
        Train the federated learning model for a round.

        Args:
            round_idx: The index of the training round.

        Returns:
            tuple: A tuple containing weights and local sample number.
        """
        weights, local_sample_num = self.trainer.train(round_idx)
        return weights, local_sample_num

    def test(self, round_idx):
        self.trainer.test(round_idx)

    def update_model(self, model_params):
        """
        Update the federated learning model with new parameters.

        Args:
            model_params: The new model parameters.
        """
        self.trainer.update_model(model_params)

    def update_dataset(self, client_index=None):
        """
        Update the dataset for training.

        Args:
            client_index: The index of the client (optional).
        """
        _client_index = client_index or self.client_index
        self.trainer.update_dataset(int(_client_index))

    def cleanup_pg(self):
        """
        Clean up the process group for distributed training.
        """
        if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            logging.info(
                "Cleaningup process group for client %s in silo %s"
                % (self.args.proc_rank_in_silo, self.args.rank_in_node)
            )
            self.process_group_manager.cleanup()
