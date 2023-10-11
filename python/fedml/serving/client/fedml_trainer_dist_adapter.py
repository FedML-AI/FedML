import logging

from fedml.constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from .fedml_trainer import FedMLTrainer
from ...ml.trainer.trainer_creator import create_model_trainer
from ...ml.engine import ml_engine_adapter


class TrainerDistAdapter:
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
        Initialize the TrainerDistAdapter.

        Args:
            args: Command-line arguments.
            device: Torch device for training.
            client_rank: Rank of the client.
            model: The neural network model.
            train_data_num: Number of training data samples.
            train_data_local_num_dict: Dictionary mapping client IDs to local training data counts.
            train_data_local_dict: Dictionary mapping client IDs to local training datasets.
            test_data_local_dict: Dictionary mapping client IDs to local test datasets.
            model_trainer: Trainer for the model.

        """
        ml_engine_adapter.model_to_device(args, model, device)

        if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            self.process_group_manager, model = ml_engine_adapter.model_ddp(args, model, device)

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
        Create and return a trainer for the federated learning process.

        Args:
            client_index: Index of the client.
            train_data_local_dict: Dictionary mapping client IDs to local training datasets.
            train_data_local_num_dict: Dictionary mapping client IDs to local training data counts.
            test_data_local_dict: Dictionary mapping client IDs to local test datasets.
            train_data_num: Number of training data samples.
            device: Torch device for training.
            args: Command-line arguments.
            model_trainer: Trainer for the model.

        Returns:
            FedMLTrainer: Trainer instance for federated learning.

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
        Perform federated training for the specified round.

        Args:
            round_idx: Index of the current training round.

        Returns:
            Tuple: A tuple containing the updated model weights and the number of local training samples.

        """
        weights, local_sample_num = self.trainer.train(round_idx)
        return weights, local_sample_num

    def update_model(self, model_params):
        """
        Update the model with new parameters.

        Args:
            model_params: Updated model parameters.

        """
        self.trainer.update_model(model_params)

    def update_dataset(self, client_index=None):
        """
        Update the local dataset for training.

        Args:
            client_index (Optional): Index of the client to update the dataset for (default is None, uses client's index).

        """
        _client_index = client_index or self.client_index
        self.trainer.update_dataset(int(_client_index))

    def cleanup_pg(self):
        """
        Clean up the process group if using distributed training.

        This method is called to clean up the process group when hierarchical federated learning is used.

        """
        if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            logging.info(
                "Cleaning up process group for client %s in silo %s"
                % (self.args.proc_rank_in_silo, self.args.rank_in_node)
            )
            self.process_group_manager.cleanup()
