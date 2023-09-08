import logging

from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedGANClientManager(FedMLCommManager):
    """
    Manager for Federated GAN client-side operations.

    Args:
        args: Configuration arguments.
        trainer: Model trainer for local training.
        comm: MPI communication object.
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        backend (str): Backend for communication (e.g., 'MPI').

    Attributes:
        trainer: Model trainer for local training.
        num_rounds: Number of communication rounds.
        args.round_idx: Current communication round index.
    """

    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """
        Start the client manager's execution.
        """
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for initialization and model updates.
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
        Handle the initialization message from the server.

        Args:
            msg_params (dict): Message parameters containing model parameters and client index.
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx = 0
        self.__train()

    def start_training(self):
        """
        Start the client training.
        """
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle the received model update message from the server.

        Args:
            msg_params (dict): Message parameters containing model parameters and client index.
        """
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            # post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        """
        Send the local model to the server.

        Args:
            receive_id (int): ID of the server receiving the model.
            weights: Model weights to be sent.
            local_sample_num: Number of local samples used for training.
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
        Perform the local training and send the updated model to the server.
        """
        logging.info("#######training########### round_id = %d" % self.args.round_idx)
        weights, local_sample_num = self.trainer.train(self.args.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)
