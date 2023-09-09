import logging
import time

from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedOptClientManager(FedMLCommManager):
    """
    Manager for Federated Optimization Clients.

    Args:
        args (object): Arguments for configuration.
        trainer (object): Trainer for client-side training.
        comm (object, optional): Communication module (default: None).
        rank (int, optional): Client's rank (default: 0).
        size (int, optional): Number of clients (default: 0).
        backend (str, optional): Backend for communication (default: "MPI").

    Attributes:
        trainer (object): Trainer for client-side training.
        num_rounds (int): Number of communication rounds.
        round_idx (int): Current communication round index.
        worker_id (int): Worker's unique identifier within the communication group.
    """

    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.worker_id = self.rank - 1

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        """
        Register handlers for receiving messages.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        """
        Handle initialization message from the server.

        Args:
            msg_params (dict): Message parameters.

        Notes:
            This method handles the initialization message from the server, including
            model parameters, average weights, and client schedule.
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        # client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.round_idx = 0
        self.__train(global_model_params, client_indexes, average_weight_dict)

    def start_training(self):
        """
        Start the training process for a new round.
        """
        self.round_idx = 0

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle the received model from the server.

        Args:
            msg_params (dict): Message parameters.

        Notes:
            This method handles the received model from the server, including model
            parameters, average weights, and client schedule. It triggers the training
            process and completes communication rounds.
        """
        logging.info("handle_message_receive_model_from_server.")

        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        # client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.round_idx += 1
        self.__train(global_model_params, client_indexes, average_weight_dict)
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, client_runtime_info):
        """
        Send the client's model to the server.

        Args:
            receive_id (int): Receiver's ID.
            weights (dict): Model parameters.
            client_runtime_info (dict): Information about client runtime.

        Notes:
            This method constructs and sends a message containing the client's model
            and runtime information to the server.
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO, client_runtime_info
        )
        self.send_message(message)


    def add_client_model(self, local_agg_model_params, model_params, weight=1.0):
        """
        Add a client's model to the local aggregated model.

        Args:
            local_agg_model_params (dict): Local aggregated model parameters.
            model_params (dict): Client's model parameters.
            weight (float, optional): Weight for the client's model (default: 1.0).

        Notes:
            This method adds client model parameters to the local aggregated model.
        """
        for name, param in model_params.items():
            if name not in local_agg_model_params:
                local_agg_model_params[name] = param * weight
            else:
                local_agg_model_params[name] += param * weight

    def __train(self, global_model_params, client_indexes, average_weight_dict):
        """
        Train the client's model.

        Args:
            global_model_params (dict): Global model parameters.
            client_indexes (list): List of client indexes.
            average_weight_dict (dict): Dictionary of average weights for clients.

        Notes:
            This method simulates client-side training, updating the local aggregated
            model with the client's contributions.
        """
        logging.info("#######training########### round_id = %d" % self.round_idx)

        local_agg_model_params = {}
        client_runtime_info = {}
        for client_index in client_indexes:
            logging.info("#######training########### Simulating client_index = %d, average weight: %f " % \
                (client_index, average_weight_dict[client_index]))
            start_time = time.time()
            self.trainer.update_model(global_model_params)
            self.trainer.update_dataset(int(client_index))
            weights, local_sample_num = self.trainer.train(self.round_idx)
            self.add_client_model(local_agg_model_params, weights,
                                weight=average_weight_dict[client_index])

            end_time = time.time()
            client_runtime = end_time - start_time
            client_runtime_info[client_index] = client_runtime
            logging.info("#######training########### End Simulating client_index = %d, consuming time: %f" % \
                (client_index, client_runtime))
        self.send_model_to_server(0, local_agg_model_params, client_runtime_info)



