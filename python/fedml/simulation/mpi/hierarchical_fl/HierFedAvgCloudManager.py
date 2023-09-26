import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message
from .utils import post_complete_message_to_sweep_process

class HierFedAVGCloudManager(FedMLCommManager):
    def __init__(
        self,
        args,
        aggregator,
        group_indexes,
        group_to_client_indexes,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        # is_preprocessed=False,
        # preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.group_indexes = group_indexes
        self.group_to_client_indexes = group_to_client_indexes
        self.round_num = args.comm_round
        self.args.round_idx = 0
        # self.is_preprocessed = is_preprocessed
        # self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def send_init_msg(self):
        # broadcast to edge servers
        global_model_params = self.aggregator.get_global_model_params()

        sampled_client_indexes = self.aggregator.client_sampling(
            self.args.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )

        sampled_group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if not group_idx in sampled_group_to_client_indexes:
                sampled_group_to_client_indexes[group_idx] = []
            sampled_group_to_client_indexes[group_idx].append(client_idx)
        logging.info(
            "client_indexes of each group = {}".format(sampled_group_to_client_indexes)
        )

        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id,
                global_model_params,
                self.group_to_client_indexes[process_id - 1],
                sampled_group_to_client_indexes[process_id - 1],
                process_id - 1
            )

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_E2C_SEND_MODEL_TO_CLOUD,
            self.handle_message_receive_model_from_edge,
        )

    def handle_message_receive_model_from_edge(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params_list = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_LIST)
        edge_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params_list, edge_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:

            global_model_params = self.aggregator.aggregate()
            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                return

            sampled_client_indexes = self.aggregator.client_sampling(
                self.args.round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

            sampled_group_to_client_indexes = {}
            for client_idx in sampled_client_indexes:
                group_idx = self.group_indexes[client_idx]
                if not group_idx in sampled_group_to_client_indexes:
                    sampled_group_to_client_indexes[group_idx] = []
                sampled_group_to_client_indexes[group_idx].append(client_idx)
            logging.info(
                "client_indexes of each group = {}".format(sampled_group_to_client_indexes)
            )

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_edge(
                    receiver_id, global_model_params,
                    sampled_group_to_client_indexes[receiver_id-1], receiver_id-1
                )

    def send_message_init_config(self, receive_id, global_model_params, total_client_indexes, sampled_client_indexed, edge_index):
        message = Message(
            MyMessage.MSG_TYPE_C2E_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_TOTAL_EDGE_CLIENTS, total_client_indexes)
        message.add_params(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS, sampled_client_indexed)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EDGE_INDEX, str(edge_index))
        self.send_message(message)

    def send_message_sync_model_to_edge(
        self, receive_id, global_model_params, sampled_client_indexed, edge_index
    ):
        logging.info("send_message_sync_model_to_edge. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_C2E_SYNC_MODEL_TO_EDGE,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS, sampled_client_indexed)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_EDGE_INDEX, str(edge_index))
        self.send_message(message)
