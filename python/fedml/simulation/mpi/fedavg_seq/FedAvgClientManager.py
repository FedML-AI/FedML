import logging
import time
from math import cos

from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.communication.message import Message
from ....core.distributed.fedml_comm_manager import FedMLCommManager




class FedAVGClientManager(FedMLCommManager):
    """
    Manager for federated learning clients using the Federated Averaging (FedAvg) algorithm.

    This class handles communication between the server and clients, as well as the training
    process on each client.

    Args:
        args (Namespace): Command-line arguments and configuration.
        trainer (object): An instance of the model trainer used for local training on clients.
        comm (object, optional): The communication backend (e.g., MPI). Defaults to None.
        rank (int, optional): The rank of the client. Defaults to 0.
        size (int, optional): The total number of clients. Defaults to 0.
        backend (str, optional): The communication backend type (e.g., MPI). Defaults to "MPI".
    """
    def __init__(
        self, args, trainer, comm=None, rank=0, size=0, backend="MPI",
    ):
        """
        Initialize the FedAVGClientManager.

        Args:
            args: The command-line arguments.
            trainer: The trainer for client-side training.
            comm: The communication backend.
            rank: The rank of the client.
            size: The total number of clients.
            backend: The communication backend (e.g., "MPI").
        """
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.worker_id = self.rank - 1

    def run(self):
        """
        Run the FedAVGClientManager.
        """
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers.
        """
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        """
        Handle initialization message from the server.

        Args:
            msg_params: The message parameters.
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
            msg_params: The message parameters.
        """
        logging.info("handle_message_receive_model_from_server.")
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
        Send the training results to the server.

        Args:
            receive_id: The ID of the recipient (server).
            weights: The model weights.
            client_runtime_info: Information about client runtime.
        """
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO, client_runtime_info)
        self.send_message(message)

    def add_client_model(self, local_agg_model_params, model_params, weight=1.0):
        """
        Add the client model parameters to the local aggregation.

        Args:
            local_agg_model_params: The local aggregation of model parameters.
            model_params: The model parameters.
            weight: The weight for averaging.
        """
        for name, param in model_params.items():
            if name not in local_agg_model_params:
                local_agg_model_params[name] = param * weight
            else:
                local_agg_model_params[name] += param * weight

    def __train(self, global_model_params, client_indexes, average_weight_dict):
        """
        Train the client model.

        Args:
            global_model_params: The global model parameters.
            client_indexes: The indexes of clients.
            average_weight_dict: The dictionary of average weights.
        """
        logging.info("#######training########### round_id = %d" % self.round_idx)

        if hasattr(self.args, "simulation_gpu_hetero"):
            simulation_gpu_hetero = self.args.simulation_gpu_hetero
            runtime_speed_ratio = self.args.gpu_hetero_ratio * self.worker_id / self.args.worker_num

        if hasattr(self.args, "simulation_environment_hetero"):
            if self.args.simulation_environment_hetero == "cos":
                runtime_speed_ratio = self.args.environment_hetero_ratio * \
                    (1 + cos(self.round_idx / self.num_rounds*3.1415926 + self.worker_id))
            else:
                raise NotImplementedError

        local_agg_model_params = {}
        client_runtime_info = {}
        for client_index in client_indexes:
            logging.info(
                "#######training########### Simulating client_index = %d, average weight: %f "
                % (client_index,
                                   average_weight_dict[client_index])
            )
            start_time = time.time()
            self.trainer.update_model(global_model_params)
            self.trainer.update_dataset(int(client_index))
            weights, local_sample_num = self.trainer.train(self.round_idx)
            self.add_client_model(local_agg_model_params, weights, weight=average_weight_dict[client_index])
            if hasattr(self.args, "simulation_gpu_hetero"):
                t_train = time.time() - start_time
                time.sleep(runtime_speed_ratio * t_train)
            end_time = time.time()
            client_runtime = end_time - start_time
            client_runtime_info[client_index] = client_runtime
            logging.info(
                "#######training########### End Simulating client_index = %d, consuming time: %f"
                % (client_index, client_runtime)
            )
        self.send_result_to_server(0, local_agg_model_params, client_runtime_info)
