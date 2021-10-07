import logging

import numpy as np
import torch

from fedml_core.distributed.client.client_manager import ClientManager
from .message_define import MyMessage

from .utils import transform_list_to_tensor
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
        list_model_params = msg_params.get_params()
        # received_model_tensor = torch.from_numpy(np.asarray(list_model_params)).float()
        received_model_tensor = transform_list_to_tensor(list_model_params)
        logging.info("handle_message_receive_model_from_server. tensor = {}".format(received_model_tensor))
        self.finish()
