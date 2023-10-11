import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class BaseCentralManager(FedMLCommManager):
    def __init__(self, args, comm, rank, size, aggregator):
        """
        Initialize the BaseCentralManager.

        Args:
            args (object): An object containing configuration parameters.
            comm (object): MPI communication object.
            rank (int): The rank of the current process.
            size (int): The total number of processes.
            aggregator (object): The aggregator for aggregating results.

        Returns:
            None
        """
        super().__init__(args, comm, rank, size)

        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """
        Run the central manager.

        This method initiates the communication with client processes and aggregates their results.

        Returns:
            None
        """
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id)
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for the central manager.

        Returns:
            None
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_INFORMATION,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        """
        Handle messages received from client processes.

        Args:
            msg_params (dict): A dictionary containing message parameters.

        Returns:
            None
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_local_result = msg_params.get(MyMessage.MSG_ARG_KEY_INFORMATION)

        self.aggregator.add_client_local_result(sender_id - 1, client_local_result)
        b_all_received = self.aggregator.check_whether_all_receive()

        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            logging.info(
                "**********************************ROUND INDEX = "
                + str(self.args.round_idx)
            )
            global_result = self.aggregator.aggregate()

            # Start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                self.finish()
                return

            for receiver_id in range(1, self.size):
                self.send_message_to_client(receiver_id, global_result)

    def send_message_init_config(self, receive_id):
        """
        Send initialization configuration message to a client process.

        Args:
            receive_id (int): The ID of the receiving client process.

        Returns:
            None
        """
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        self.send_message(message)

    def send_message_to_client(self, receive_id, global_result):
        """
        Send a message to a client process containing global results.

        Args:
            receive_id (int): The ID of the receiving client process.
            global_result (object): The global result to be sent.

        Returns:
            None
        """
        message = Message(
            MyMessage.MSG_TYPE_S2C_INFORMATION, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_INFORMATION, global_result)
        self.send_message(message)
