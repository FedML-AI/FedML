import logging

from .message_define import MyMessage
from ....core.distributed.communication.message import Message
from ....core.distributed.fedml_comm_manager import FedMLCommManager

class DecentralizedWorkerManager(FedMLCommManager):
    """
    Class representing a decentralized federated learning worker in a distributed system.
    """
    def __init__(self, args, comm, rank, size, trainer, topology_manager):
        """
        Manages decentralized federated learning workers in a distributed system.

        Args:
            args: Configuration arguments.
            comm: MPI communication object for distributed communication.
            rank: The rank (ID) of the current worker.
            size: The total number of workers in the distributed system.
            trainer: The decentralized worker/trainer.
            topology_manager: The topology manager for communication between workers.
        """
        super().__init__(args, comm, rank, size)
        self.worker_index = rank
        self.trainer = trainer
        self.topology_manager = topology_manager
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        """
        Start the training process for decentralized federated learning workers.
        """
        self.start_training()
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for handling incoming messages.
        """
        self.register_message_receive_handler(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.handle_msg_from_neighbor)

    def start_training(self):
        """
        Initialize and start the training process.
        """
        self.round_idx = 0
        self.__train()

    def handle_msg_from_neighbor(self, msg_params):
        """
        Handle messages received from neighboring workers.

        Args:
            msg_params: Parameters included in the received message.
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        training_iteration_result = msg_params.get(MyMessage.MSG_ARG_KEY_PARAMS_1)
        logging.info("handle_msg_from_neighbor. sender_id = " + str(sender_id))
        self.trainer.add_result(sender_id, training_iteration_result)
        if self.trainer.check_whether_all_receive():
            logging.info(">>>>>>>>>>>>>>>WORKER %d, ROUND %d finished!<<<<<<<<" % (self.worker_index, self.round_idx))
            self.round_idx += 1
            if self.round_idx == self.num_rounds:
                self.finish()
            self.__train()

    def __train(self):
        """
        Perform the training process and communicate with neighboring workers.
        """
        # Perform the training process here (e.g., training iteration)
        training_iteration_result = self.trainer.train()

        # Send the training iteration result to neighboring workers
        for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.worker_index):
            self.send_result_to_neighbors(neighbor_idx, training_iteration_result)

    def send_message_init_config(self, receive_id):
        """
        Send an initialization message to a specified worker.

        Args:
            receive_id: The ID of the receiving worker.
        """
        message = Message(MyMessage.MSG_TYPE_INIT, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_result_to_neighbors(self, receive_id, client_params1):
        """
        Send training iteration results to neighboring workers.

        Args:
            receive_id: The ID of the receiving worker.
            client_params1: Parameters to be sent in the message.
        """
        logging.info("send_result_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PARAMS_1, client_params1)
        self.send_message(message)
