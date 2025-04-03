import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message
from .utils import post_complete_message_to_sweep_process


class HierFedAVGEdgeManager(FedMLCommManager):
    def __init__(
        self,
        group,
        args,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
    ):
        super().__init__(args, comm, rank, size, backend)
        self.num_rounds = args.comm_round
        self.args.round_idx = 0
        self.group =group

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2E_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2E_SYNC_MODEL_TO_EDGE,
            self.handle_message_receive_model_from_cloud,
        )

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        total_client_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_TOTAL_EDGE_CLIENTS)
        sampled_client_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS)
        total_sampled_data_size = msg_params.get(MyMessage.MSG_ARG_KEY_TOTAL_SAMPLED_DATA_SIZE)
        edge_index = msg_params.get(MyMessage.MSG_ARG_KEY_EDGE_INDEX)

        self.group.setup_clients(total_client_indexes)
        self.args.round_idx = 0
        w_group_list, sample_num_list = self.group.train(self.args.round_idx, global_model_params,
                                                         sampled_client_indexes, total_sampled_data_size)

        self.send_model_to_cloud(0, w_group_list, sample_num_list)

    def handle_message_receive_model_from_cloud(self, msg_params):
        logging.info("handle_message_receive_model_from_cloud.")
        sampled_client_indexes = msg_params.get(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS)
        total_sampled_data_size = msg_params.get(MyMessage.MSG_ARG_KEY_TOTAL_SAMPLED_DATA_SIZE)
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        edge_index = msg_params.get(MyMessage.MSG_ARG_KEY_EDGE_INDEX)

        self.args.round_idx += 1
        w_group_list, sample_num_list = self.group.train(self.args.round_idx, global_model_params,
                                                         sampled_client_indexes, total_sampled_data_size)
        self.send_model_to_cloud(0, w_group_list, sample_num_list)

        if self.args.round_idx == self.num_rounds:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_cloud(self, receive_id, w_group_list, edge_sample_num):
        message = Message(
            MyMessage.MSG_TYPE_E2C_SEND_MODEL_TO_CLOUD,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_LIST, w_group_list)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, edge_sample_num)
        self.send_message(message)
