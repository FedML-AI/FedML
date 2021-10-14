import logging
import os
import sys

import torch

from .message_define import MyMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager


class RPCServerManager(ServerManager):
    def __init__(self, args, comm=None, rank=0, size=0, backend="GRPC"):
        super().__init__(args, comm, rank=rank, size=size, backend=backend)
        self.args = args

    def run(self):
        super().run()

    def send_model_params(self):
        global_model_params = torch.randn(5000, 5000)
        # global_model_params = torch.randn(5000, 5000)
        logging.info("send_model_params START")
        if self.args.backend == "GRPC":
            global_model_params = global_model_params.detach()
        self.send_message_model_params_to_client(1, global_model_params)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        logging.info("handle_message_receive_model_from_client = {}".format(msg_params))

    def send_message_model_params_to_client(self, receive_id, global_model_params):
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)
