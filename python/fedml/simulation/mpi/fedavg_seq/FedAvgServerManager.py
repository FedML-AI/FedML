import logging
import time

import wandb

from .message_define import MyMessage
from .utils import transform_tensor_to_list
from ....core.distributed.communication.message import Message
from ....core.distributed.fedml_comm_manager import FedMLCommManager


class FedAVGServerManager(FedMLCommManager):
    """
    Class responsible for managing the server in the FedAVG federated learning system.

    Args:
        args (Namespace): Command-line arguments and configuration.
        aggregator (object): An instance of the aggregator used for federated learning.
        comm (object, optional): The communication backend (e.g., MPI). Defaults to None.
        rank (int, optional): The rank of the server process. Defaults to 0.
        size (int, optional): The total number of processes. Defaults to 0.
        backend (str, optional): The backend used for communication. Defaults to "MPI".
        is_preprocessed (bool, optional): Indicates whether client lists are preprocessed. Defaults to False.
        preprocessed_client_lists (list, optional): Preprocessed client lists. Defaults to None.
    """
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        """
        Run the server manager to coordinate federated learning.

        This method runs the server manager to coordinate the federated learning process.
        """
        super().run()

    def send_init_msg(self):
        """
        Send the initialization message to clients.

        This method sends an initialization message to client processes to begin federated learning.
        """
        # sampling clients
        self.previous_time = time.time()
        client_indexes = self.aggregator.client_sampling(
            self.args.round_idx, self.args.client_num_in_total, self.args.client_num_per_round,
        )

        client_schedule = self.aggregator.generate_client_schedule(self.args.round_idx, client_indexes)
        average_weight_dict = self.aggregator.get_average_weight(client_indexes)

        global_model_params = self.aggregator.get_global_model_params()

        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, average_weight_dict, client_schedule)

    def register_message_receive_handlers(self):
        """
        Register message receive handlers.

        This method registers message receive handlers for processing incoming messages.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        """
        Handle the received model update from a client.

        Args:
            msg_params (dict): The parameters of the received message.

        This method handles the model update received from a client during federated learning.
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        client_runtime_info = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO)
        self.aggregator.record_client_runtime(sender_id - 1, client_runtime_info)

        # self.aggregator.add_local_trained_result(
        #     sender_id - 1, model_params, local_sample_number
        # )
        self.aggregator.add_local_trained_result(sender_id - 1, model_params)

        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.enable_wandb:
                wandb.log({"RunTimeOneRound": time.time() - self.previous_time, "round": self.args.round_idx})
                self.previous_time = time.time()

            global_model_params = self.aggregator.aggregate()
            current_time = time.time()
            self.aggregator.test_on_server_for_all_clients(self.args.round_idx)
            if self.args.enable_wandb:
                wandb.log({"TestTimeOneRound": time.time() - current_time, "round": self.args.round_idx})

            # Exclude the time of Testing 
            self.previous_time = time.time()

            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                self.finish()
                print("here")
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.args.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.args.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.args.round_idx, self.args.client_num_in_total, self.args.client_num_per_round,
                )
            client_schedule = self.aggregator.generate_client_schedule(self.args.round_idx, client_indexes)
            average_weight_dict = self.aggregator.get_average_weight(client_indexes)

            global_model_params = self.aggregator.get_global_model_params()

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, average_weight_dict, client_schedule
                )

    def send_message_init_config(self, receive_id, global_model_params, average_weight_dict, client_schedule):
        """
        Send the initialization configuration message to a client.

        Args:
            receive_id (int): The ID of the receiving client.
            global_model_params (dict): Global model parameters.
            average_weight_dict (dict): Average weight dictionary for clients.
            client_schedule (list): The schedule of clients for the current round.

        This method sends an initialization configuration message to a client process.
        """
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        # message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS, average_weight_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE, client_schedule)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, average_weight_dict, client_schedule):
        """
        Send the model synchronization message to a client.

        Args:
            receive_id (int): The ID of the receiving client.
            global_model_params (dict): Global model parameters.
            average_weight_dict (dict): Average weight dictionary for clients.
            client_schedule (list): The schedule of clients for the current round.

        This method sends a model synchronization message to a client process.
        """
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        # message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS, average_weight_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE, client_schedule)
        self.send_message(message)
