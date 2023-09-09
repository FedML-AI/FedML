import logging

import torch.distributed as dist


class ClientSlaveManager:
    def __init__(self, args, trainer_dist_adapter):
        """
        Initialize the ClientSlaveManager.

        Args:
            args: Command-line arguments.
            trainer_dist_adapter: Trainer distributed adapter.

        """
        self.trainer_dist_adapter = trainer_dist_adapter
        self.args = args
        self.round_idx = 0
        self.num_rounds = args.comm_round
        self.finished = False

    def train(self):
        """
        Perform training for the current round.

        This method synchronizes with the process group, updates the model and dataset if necessary, and initiates training
        for the current round.

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
        Finish the training process.

        This method performs cleanup operations and logs the completion of training.

        """
        self.trainer_dist_adapter.cleanup_pg()
        logging.info(
            "Training finished for slave client rank %s in silo %s"
            % (self.args.proc_rank_in_silo, self.args.rank_in_node)
        )
        self.finished = True

    def await_sync_process_group(self, src=0):
        """
        Wait for synchronization with the process group.

        Args:
            src: Source rank for synchronization (default is 0).

        Returns:
            List: A list containing round_idx, model_params, and client_index.

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
        Start the client manager's main execution loop.

        This method continuously trains the client while it is not finished.

        """
        while not self.finished:
            self.train()
