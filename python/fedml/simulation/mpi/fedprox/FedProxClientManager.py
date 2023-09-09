import logging

from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedProxClientManager(FedMLCommManager):
    """
    Client manager for Federated Proximal training.

    Args:
        args (object): Arguments for configuration.
        trainer (object): Trainer for the client.
        comm (object): Communication object.
        rank (int): Rank of the client.
        size (int): Total number of participants.
        backend (str): Backend for communication (default: "MPI").
    """

    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for the client manager.

        This method registers message handlers for receiving initialization
        and model synchronization messages.

        Message Types:
        - MyMessage.MSG_TYPE_S2C_INIT_CONFIG
        - MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT
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
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx = 0
        self.__train()

    def start_training(self):
        """
        Start the training process.
        """
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle model synchronization message from the server.

        Args:
            msg_params (dict): Message parameters.
        """
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        """
        Send the trained model to the server.

        Args:
            receive_id (int): Receiver ID (typically the server).
            weights (object): Model weights.
            local_sample_num (int): Number of local training samples.
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        """
        Execute the training process.
        """
        logging.info("#######training########### round_id = %d" % self.args.round_idx)
        weights, local_sample_num = self.trainer.train(self.args.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)
