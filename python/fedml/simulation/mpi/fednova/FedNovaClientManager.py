import logging
import time


from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedNovaClientManager(FedMLCommManager):
    """
    Manager for the client-side of the FedNova federated learning process.

    Parameters:
        args: Command-line arguments.
        trainer: Client trainer responsible for local training.
        comm: Communication backend for distributed training.
        rank (int): Rank of the client process.
        size (int): Total number of processes.
        backend (str): Communication backend (e.g., "MPI").

    Methods:
        __init__: Initialize the FedNovaClientManager.
        run: Start the client manager.
        register_message_receive_handlers: Register message receive handlers for handling incoming messages.
        handle_message_init: Handle the initialization message received from the server.
        start_training: Start the training process.
        handle_message_receive_model_from_server: Handle the received model from the server.
        send_result_to_server: Send training results to the server.
        add_client_model: Add client model parameters to the aggregation.
        __train: Perform the training process for the specified clients.
    """

    def __init__(
        self,
        args,
        trainer,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
    ):
        """
        Initialize the FedNovaClientManager.

        Args:
            args: Command-line arguments.
            trainer: Client trainer responsible for local training.
            comm: Communication backend for distributed training.
            rank (int): Rank of the client process.
            size (int): Total number of processes.
            backend (str): Communication backend (e.g., "MPI").
        """
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.worker_id = self.rank - 1

    def run(self):
        """
        Start the client manager.
        """
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for handling incoming messages.
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
        Handle the initialization message received from the server.

        Args:
            msg_params: Parameters included in the received message.
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.round_idx = 0
        self.__train(global_model_params, client_indexes, average_weight_dict)

    def start_training(self):
        """
        Start the training process.
        """
        self.round_idx = 0

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle the received model from the server.

        Args:
            msg_params: Parameters included in the received message.
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.round_idx += 1
        self.__train(global_model_params, client_indexes, average_weight_dict)
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def send_result_to_server(self, receive_id, weights, client_runtime_info):
        """
        Send training results to the server.

        Args:
            receive_id: ID of the recipient (e.g., the server).
            weights: Model weights or parameters.
            client_runtime_info: Information about client runtime.
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO, client_runtime_info)
        self.send_message(message)

    def add_client_model(self, local_agg_model_params, client_index, grad, t_eff, weight=1.0):
        """
        Add client model parameters to the aggregation.

        Args:
            local_agg_model_params: Local aggregation of model parameters.
            client_index: Index or ID of the client.
            grad: Gradients computed during training.
            t_eff: Efficiency factor.
            weight: Weight assigned to the client's contribution.
        """
        local_agg_model_params.append({
            "grad": grad, "t_eff": t_eff,
        })

    def __train(self, global_model_params, client_indexes, average_weight_dict):
        """
        Perform the training process for the specified clients.

        Args:
            global_model_params: Global model parameters.
            client_indexes: Indexes of the clients to train.
            average_weight_dict: Dictionary of average weights for clients.
        """
        logging.info("#######training########### round_id = %d" % self.round_idx)

        local_agg_model_params = []
        client_runtime_info = {}
        for client_index in client_indexes:
            logging.info("#######training########### Simulating client_index = %d, average weight: %f " % \
                (client_index, average_weight_dict[client_index]))
            start_time = time.time()
            self.trainer.update_model(global_model_params)
            self.trainer.update_dataset(int(client_index))
            loss, grad, t_eff = self.trainer.train(self.round_idx)
            self.add_client_model(local_agg_model_params, client_index, grad, t_eff,
                                weight=average_weight_dict[client_index])

            end_time = time.time()
            client_runtime = end_time - start_time
            client_runtime_info[client_index] = client_runtime
            logging.info("#######training########### End Simulating client_index = %d, consuming time: %f" % \
                (client_index, client_runtime))
        self.send_result_to_server(0, local_agg_model_params, client_runtime_info)
