import logging

import torch.distributed as dist


class ClientSlaveManager:
    """
    Manages the training process for a federated learning client slave.

    Args:
        args (object): An object containing client configuration parameters.
        trainer_dist_adapter: An instance of the trainer distribution adapter.

    Attributes:
        trainer_dist_adapter: An instance of the trainer distribution adapter.
        args (object): An object containing client configuration parameters.
        round_idx (int): The current training round index.
        num_rounds (int): The total number of training rounds.
        finished (bool): A flag indicating if training has finished.

    Methods:
        train():
            Perform training for the current round.
        finish():
            Finish the client slave's training.
        await_sync_process_group(src=0):
            Await synchronization with the process group and receive round information.
        run():
            Start the client slave's training process.

    """
    def __init__(self, args, trainer_dist_adapter):
        """
        Initialize the ClientSlaveManager.

        Args:
            args (object): An object containing client configuration parameters.
            trainer_dist_adapter: An instance of the trainer distribution adapter.

        Returns:
            None
        """
        self.trainer_dist_adapter = trainer_dist_adapter
        self.args = args
        self.round_idx = 0
        self.num_rounds = args.comm_round
        self.finished = False

    def train(self):
        """
        Perform training for the current round.

        Returns:
            None
        """
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
        Finish the client slave's training.

        Returns:
            None
        """
        self.trainer_dist_adapter.cleanup_pg()
        logging.info(
            "Training finished for slave client rank %s in silo %s"
            % (self.args.proc_rank_in_silo, self.args.rank_in_node)
        )
        self.finished = True

    def await_sync_process_group(self, src=0):
        """
        Await synchronization with the process group and receive round information.

        Args:
            src (int): The source process rank to receive data from (default is 0).

        Returns:
            list: A list containing round index, model parameters, and client index.
        """
        logging.info("process %d waiting for round number" % dist.get_rank())
        objects = [None, None, None]
        dist.broadcast_object_list(
            objects, src=src, group=self.trainer_dist_adapter.process_group_manager.get_process_group(),
        )
        logging.info("process {} received round_number {}".format(dist.get_rank(), objects[0]))
        return objects

    def run(self):
        """
        Start the client slave's training process.

        Returns:
            None
        """
        while not self.finished:
            self.train()
