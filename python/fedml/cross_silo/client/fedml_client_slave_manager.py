import logging

import torch.distributed as dist

from fedml.constants import FEDML_CROSS_SILO_CUSTOMIZED_HIERARCHICAL_KEY
from .utils import check_method_override


class ClientSlaveManager:
    def __init__(self, args, trainer_dist_adapter):
        """
        Initialize a federated learning client manager for a slave.

        Args:
            args: The command-line arguments.
            trainer_dist_adapter: The trainer distributed adapter.
        """
        self.trainer_dist_adapter = trainer_dist_adapter
        self.args = args
        self.round_idx = 0
        self.num_rounds = args.comm_round
        self.finished = False

        if self.use_customized_hierarchical:
            trainer_class_name = self.trainer_dist_adapter.trainer.trainer.__class__.__name__

            if not self.has_customized_await_sync_process_group:
                raise RuntimeError(
                    f"\"await_sync_process_group\" implementation is required for class {trainer_class_name}"
                    f" for customized hierarchical cross-silo."
                )

            if not self.has_customized_cleanup_process_group:
                logging.warning(
                    f"\"cleanup_process_group\" implementation is not provided for class {trainer_class_name}"
                    f" for customized hierarchical cross-silo."
                )

    @property
    def use_customized_hierarchical(self) -> bool:
        """
        Determine whether customized hierarchical cross-silo is enabled.

        Returns:
            bool: True if customized hierarchical cross-silo is enabled, False otherwise.
        """
        return getattr(self.args, FEDML_CROSS_SILO_CUSTOMIZED_HIERARCHICAL_KEY, False)

    @property
    def has_customized_await_sync_process_group(self) -> bool:
        """
        Check if the trainer has a customized "await_sync_process_group" method.

        Returns:
            bool: True if the method is overridden, False otherwise.
        """
        return check_method_override(
            cls_obj=self.trainer_dist_adapter.trainer.trainer,
            method_name="await_sync_process_group"
        )

    @property
    def has_customized_cleanup_process_group(self) -> bool:
        """
        Check if the trainer has a customized "cleanup_process_group" method.

        Returns:
            bool: True if the method is overridden, False otherwise.
        """
        return check_method_override(
            cls_obj=self.trainer_dist_adapter.trainer.trainer,
            method_name="cleanup_process_group"
        )

    def train(self):
        """
        Perform a training round for the federated learning client.
        """
        if self.use_customized_hierarchical:
            [round_idx, model_params, client_index] = self.customized_await_sync_process_group()
        else:
            [round_idx, model_params, client_index] = self.await_sync_process_group()
        if round_idx:
            self.round_idx = round_idx
        if model_params:
            self.trainer_dist_adapter.update_model(model_params)
        if client_index:
            self.trainer_dist_adapter.update_dataset(int(client_index))

        if self.round_idx == self.num_rounds:
            logging.info("Finishing Client Slave")
            self.finish()
            return

        self.trainer_dist_adapter.train(self.round_idx)

    def finish(self):
        """
        Finish the federated learning client's training process.
        """
        if self.use_customized_hierarchical:
            self.customized_cleanup_process_group()
        else:
            self.trainer_dist_adapter.cleanup_pg()
            logging.info(
                "Training finished for slave client rank %s in silo %s"
                % (self.args.proc_rank_in_silo, self.args.rank_in_node)
            )
        self.finished = True

    def await_sync_process_group(self, src: int = 0) -> list:
        """
        Await synchronization of the process group.

        Args:
            src (int): The source rank for synchronization.

        Returns:
            list: A list containing round number, model parameters, and client index.
        """

        logging.info("process %d waiting for round number" % dist.get_rank())
        objects = [None, None, None]
        dist.broadcast_object_list(
            objects, src=src, group=self.trainer_dist_adapter.process_group_manager.get_process_group(),
        )
        logging.info("process {} received round_number {}".format(dist.get_rank(), objects[0]))
        return objects

    def customized_await_sync_process_group(self, src: int = 0) -> list:
        """
        Perform a customized await synchronization of the process group.

        Args:
            src (int): The source rank for synchronization.

        Returns:
            list: A list containing round number, model parameters, and client index.
        """
        trainer = self.trainer_dist_adapter.trainer.trainer
        trainer_class_name = trainer.__class__.__name__

        if not self.has_customized_await_sync_process_group:
            raise RuntimeError(
                f"\"await_sync_process_group\" implementation is required for class {trainer_class_name}"
                f" for customized hierarchical cross-silo."
            )

        return trainer.await_sync_process_group(src)

    def customized_cleanup_process_group(self) -> None:
        """
        Perform a customized cleanup of the process group.
        """
        trainer = self.trainer_dist_adapter.trainer.trainer
        if self.has_customized_cleanup_process_group:
            trainer.cleanup_process_group()

    def run(self):
        """
        Run the federated learning client manager.
        """
        while not self.finished:
            self.train()
