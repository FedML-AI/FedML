import logging

from fedml_core.distributed.client.client_manager import ClientManager
from .message_define import MyMessage


class RPCClientManager(ClientManager):
    def __init__(self, args, comm=None, rank=0, size=0, backend="GRPC"):
        super().__init__(args, comm, rank=rank, size=size, backend=backend)
        self.args = args

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server
        )

    def handle_message_receive_model_from_server(self, msg_params):
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        logging.info("handle_message_receive_model_from_server. tensor.shape = {}".format(model_params.shape))
