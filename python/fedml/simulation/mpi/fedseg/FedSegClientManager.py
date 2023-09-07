import logging

from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedSegClientManager(FedMLCommManager):
    """
    Client manager for federated segmentation.

    This class manages the client-side communication and training in a federated segmentation system.

    Args:
        args: Additional configuration arguments.
        trainer: Model trainer for federated segmentation.
        comm: MPI communicator for distributed communication.
        rank (int): Rank of the client.
        size (int): Total number of processes.
        backend (str): Communication backend (default: "MPI").

    Attributes:
        args: Additional configuration arguments.
        trainer: Model trainer for federated segmentation.
        num_rounds (int): Number of communication rounds.

    Methods:
        run(): Start the client manager.
        register_message_receive_handlers(): Register message handlers for receiving initialization and model synchronization messages.
        handle_message_init(msg_params): Handle the initialization message from the central server.
        start_training(): Start the training process.
        handle_message_receive_model_from_server(msg_params): Handle received model updates from the central server.
        send_model_to_server(receive_id, weights, local_sample_num, train_evaluation_metrics, test_evaluation_metrics): Send trained model updates to the central server.
    """
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        """
        Initialize the FedSegClientManager.

        Args:
            args: Additional configuration arguments.
            trainer: Model trainer for federated segmentation.
            comm: MPI communicator for distributed communication.
            rank (int): Rank of the client.
            size (int): Total number of processes.
            backend (str): Communication backend (default: "MPI").
        """
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """
        Start the client manager.

        Notes:
            This function starts the client manager to handle communication and training.
        """
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message handlers for receiving initialization and model synchronization messages.

        Notes:
            This function registers message handlers to process incoming messages from the central server.
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
        Handle the initialization message from the central server.

        Args:
            msg_params (dict): Parameters included in the received message.

        Notes:
            This function processes the initialization message from the central server, updates the model and dataset, and starts training.
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info(
            "Client {0} received global model params from central server".format(
                client_index
            )
        )
        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx = 0
        self.__train()

    def start_training(self):
        """
        Start the training process.

        Notes:
            This function initiates the training process on the client side.
        """
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle received model updates from the central server.

        Args:
            msg_params (dict): Parameters included in the received message.

        Notes:
            This function processes received model updates from the central server, updates the model and dataset, and continues training.
        """
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(
        self,
        receive_id,
        weights,
        local_sample_num,
        train_evaluation_metrics,
        test_evaluation_metrics,
    ):
        """
        Send trained model updates to the central server.

        Args:
            receive_id (int): Receiver's ID.
            weights: Trained model parameters.
            local_sample_num (int): Number of local training samples.
            train_evaluation_metrics: Evaluation metrics for training.
            test_evaluation_metrics: Evaluation metrics for testing.

        Notes:
            This function sends the trained model updates and evaluation metrics to the central server.
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(
            MyMessage.MSG_ARG_KEY_TRAIN_EVALUATION_METRICS, train_evaluation_metrics
        )
        message.add_params(
            MyMessage.MSG_ARG_KEY_TEST_EVALUATION_METRICS, test_evaluation_metrics
        )
        self.send_message(message)

    def __train(self):
        """
        Perform training on the client side.

        Notes:
            This method initiates the training process on the client side, including testing the global parameters, training the local model, and sending updates to the central server.
        """

        train_evaluation_metrics = test_evaluation_metrics = None
        logging.info(
            "####### Testing Global Params ########### round_id = {}".format(
                self.args.round_idx
            )
        )
        train_evaluation_metrics, test_evaluation_metrics = self.trainer.test()
        logging.info(
            "####### Training ########### round_id = {}".format(self.args.round_idx)
        )
        weights, local_sample_num = self.trainer.train()
        self.send_model_to_server(
            0,
            weights,
            local_sample_num,
            train_evaluation_metrics,
            test_evaluation_metrics,
        )
