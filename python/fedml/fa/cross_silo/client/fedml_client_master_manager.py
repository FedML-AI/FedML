import json
import logging
import platform
import time
from fedml import mlops
from fedml.core import FedMLCommManager
from fedml.core.distributed.communication.message import Message
from fedml.core.mlops import MLOpsProfilerEvent
from .message_define import MyMessage
import torch.distributed as dist


class ClientMasterManager(FedMLCommManager):
    """
    Manages the communication and training process for a federated learning client master.

    Args:
        args (object): An object containing client configuration parameters.
        trainer_dist_adapter: An instance of the trainer distribution adapter.
        comm: A communication backend (default is None).
        rank (int): The rank of the client (default is 0).
        size (int): The size of the communication group (default is 0).
        backend (str): The communication backend (default is "MPI").

    Attributes:
        trainer_dist_adapter: An instance of the trainer distribution adapter.
        args (object): An object containing client configuration parameters.
        num_rounds (int): The total number of communication rounds.
        round_idx (int): The current communication round index.
        rank (int): The rank of the client.
        client_real_ids (list): A list of client real IDs.
        client_real_id (str): The client's real ID.
        has_sent_online_msg (bool): A flag indicating if the online message has been sent.

    Methods:
        register_message_receive_handlers():
            Register message receive handlers for various message types.
        handle_message_connection_ready(msg_params):
            Handle the connection-ready message.
        handle_message_check_status(msg_params):
            Handle the check-client-status message.
        handle_message_init(msg_params):
            Handle the initialization message.
        handle_message_receive_model_from_server(msg_params):
            Handle the message to receive a model from the server.
        handle_message_finish(msg_params):
            Handle the message indicating the completion of training.
        cleanup():
            Perform cleanup after training finishes.
        send_model_to_server(receive_id, weights, local_sample_num):
            Send the model and related information to the server.
        send_client_status(receive_id, status="ONLINE"):
            Send the client's status to the server.
        report_training_status(status):
            Report the training status to MLOps.
        sync_process_group(round_idx, model_params=None, client_index=None, src=0):
            Synchronize the process group with round information.
        __train():
            Perform the training for the current round.
        run():
            Start the client master manager's communication and training process.

    """
    def __init__(self, args, trainer_dist_adapter, comm=None, rank=0, size=0, backend="MPI"):
        """
        Initialize the ClientMasterManager.

        Args:
            args (object): An object containing client configuration parameters.
            trainer_dist_adapter: An instance of the trainer distribution adapter.
            comm: A communication backend (default is None).
            rank (int): The rank of the client (default is 0).
            size (int): The size of the communication group (default is 0).
            backend (str): The communication backend (default is "MPI").

        Returns:
            None
        """
        super().__init__(args, comm, rank, size, backend)
        self.trainer_dist_adapter = trainer_dist_adapter
        self.args = args
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.rank = rank
        self.client_real_ids = json.loads(args.client_id_list)
        logging.info("self.client_real_ids = {}".format(self.client_real_ids))
        # For the client, len(self.client_real_ids)==1: we only specify its client id in the list, not including others.
        self.client_real_id = self.client_real_ids[0]
        self.has_sent_online_msg = False

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for various message types.

        Returns:
            None
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.handle_message_check_status
        )

        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_FINISH, self.handle_message_finish,
        )

    def handle_message_connection_ready(self, msg_params):
        """
        Handle the connection-ready message.

        Args:
            msg_params (dict): Message parameters.

        Returns:
            None
        """
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(0)

            mlops.log_sys_perf(self.args)

    def handle_message_check_status(self, msg_params):
        """
        Handle the check-client-status message.

        Args:
            msg_params (dict): Message parameters.

        Returns:
            None
        """
        self.send_client_status(0)

    def handle_message_init(self, msg_params):
        """
        Handle the initialization message.

        Args:
            msg_params (dict): Message parameters.

        Returns:
            None
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        data_silo_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        init_msg = msg_params.get(MyMessage.MSG_INIT_MSG_TO_CLIENTS)
        logging.info("data_silo_index = %s" % str(data_silo_index))

        # Notify MLOps with training status.
        self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING)

        self.trainer_dist_adapter.update_dataset(int(data_silo_index))
        self.trainer_dist_adapter.set_server_data(global_model_params)
        self.trainer_dist_adapter.set_init_msg(init_msg)
        self.round_idx = 0

        self.__train()
        self.round_idx += 1

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle the message to receive a model from the server.

        Args:
            msg_params (dict): Message parameters.

        Returns:
            None
        """
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer_dist_adapter.update_dataset(int(client_index))
        logging.info("current round index {}, total rounds {}".format(self.round_idx, self.num_rounds))
        self.trainer_dist_adapter.set_server_data(model_params)
        if self.round_idx < self.num_rounds:
            self.__train()
            self.round_idx += 1

    def handle_message_finish(self, msg_params):
        """
        Handle the message indicating the completion of training.

        Args:
            msg_params (dict): Message parameters.

        Returns:
            None
        """
        logging.info(" ====================cleanup ====================")
        self.cleanup()

    def cleanup(self):
        """
        Perform cleanup after training finishes.

        Returns:
            None
        """
        self.finish()
        mlops.log_training_finished_status()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        """
        Send the model and related information to the server.

        Args:
            receive_id (int): The ID of the receiver.
            weights (object): Model weights or parameters.
            local_sample_num (int): The number of local samples.

        Returns:
            None
        """
        tick = time.time()
        mlops.event("comm_c2s", event_started=True, event_value=str(self.round_idx))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.client_real_id, receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

        MLOpsProfilerEvent.log_to_wandb({"Communication/Send_Total": time.time() - tick})
        mlops.log_client_model_info(
            self.round_idx+1, self.num_rounds, model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def send_client_status(self, receive_id, status="ONLINE"):
        """
        Send the client's status to the server.

        Args:
            receive_id (int): The ID of the receiver.
            status (str): The client's status (default is "ONLINE").

        Returns:
            None
        """
        logging.info("send_client_status")
        logging.info("self.client_real_id = {}".format(self.client_real_id))
        message = Message(MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.client_real_id, receive_id)
        sys_name = platform.system()
        if sys_name == "Darwin":
            sys_name = "Mac"
        # Debug for simulation mobile system
        # sys_name = MyMessage.MSG_CLIENT_OS_ANDROID

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, sys_name)
        self.send_message(message)

    def report_training_status(self, status):
        """
        Report the training status to MLOps.

        Args:
            status (str): The training status to report.

        Returns:
            None
        """
        mlops.log_training_status(status)

    def sync_process_group(self, round_idx, model_params=None, client_index=None, src=0):
        """
        Synchronize the process group with round information.

        Args:
            round_idx (int): The current round index.
            model_params (object): Model weights or parameters (default is None).
            client_index (int): The index of the client (default is None).
            src (int): The source process rank (default is 0).

        Returns:
            None
        """
        logging.info("sending round number to pg")
        round_number = [round_idx, model_params, client_index]
        dist.broadcast_object_list(
            round_number, src=src, group=self.trainer_dist_adapter.process_group_manager.get_process_group(),
        )
        logging.info("round number %d broadcast to process group" % round_number[0])

    def __train(self):
        """
        Perform the training for the current round.

        Returns:
            None
        """
        logging.info("#######training########### round_id = %d" % self.round_idx)

        mlops.event("train", event_started=True, event_value=str(self.round_idx))

        client_submission, local_sample_num = self.trainer_dist_adapter.local_analyze(self.round_idx)

        mlops.event("train", event_started=False, event_value=str(self.round_idx))

        self.send_model_to_server(0, client_submission, local_sample_num)

    def run(self):
        """
        Start the client master manager's communication and training process.

        Returns:
            None
        """
        super().run()
