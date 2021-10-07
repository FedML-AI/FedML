import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))

from fedml_core.distributed.client.client_manager import ClientManager

from .message_define import MyMessage


class RPCClientManager(ClientManager):
    def __init__(self, args, comm=None, rank=0, size=0, backend="GRPC"):
        super().__init__(args, comm, rank, size, backend)

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server
        )

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server. msg_params = {}".format(msg_params))
        self.finish()

