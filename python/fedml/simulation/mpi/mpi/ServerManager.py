import logging
import time

import wandb

from .message_define import MyMessage
from .utils import transform_tensor_to_list
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message

from fedml.ml.ml_message import MLMessage
from fedml.ml.trainer.local_cache import FedMLLocalCache


class ServerManager(FedMLCommManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        FedMLLocalCache.init(args, root=args.local_cache_root)
        self.local_cache_path = FedMLLocalCache.path
        if not self.args.client_num_per_round == (size - 1):
            logging.info(f"ERROR: client_num_per_round {self.args.client_num_per_round} is \
                not equal to number of workers {size - 1}, Please set them equal, or use mpi seq mode")
            self.finish()

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        self.previous_time = time.time()
        client_indexes = self.aggregator.client_sampling(
            self.args.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )
        # global_model_params = self.aggregator.get_global_model_params()
        server_result = self.aggregator.get_init_server_result()
        server_result.add(MLMessage.SAMPLE_NUM_DICT, dict([
            (client_index, self.aggregator.train_data_local_num_dict[client_index]) for client_index in client_indexes
        ]))
        server_result.add(MLMessage.GLOBAL_ROUND, self.args.round_idx)
        logging.info(f"client_indexes = {client_indexes}")

        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, server_result, client_indexes[process_id - 1]
            )

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        # client_result = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_RESULT)
        client_result = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            sender_id - 1, client_result, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.enable_wandb:
                wandb.log({"RunTimeOneRound": time.time() - self.previous_time, "round": self.args.round_idx})
                self.previous_time = time.time()
            # global_model_params = self.aggregator.aggregate()
            server_result = self.aggregator.aggregate()
            current_time = time.time()
            self.aggregator.test_on_server_for_all_clients(self.args.round_idx)
            if self.args.enable_wandb:
                wandb.log({"TestTimeOneRound": time.time() - current_time, "round": self.args.round_idx})

            self.previous_time = time.time()

            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                FedMLLocalCache.finalize(self.args)
                self.finish()
                print("here")
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.args.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.args.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.args.round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round,
                )
            server_result.add(MLMessage.SAMPLE_NUM_DICT, dict([
                (client_index, self.aggregator.train_data_local_num_dict[client_index]) for client_index in client_indexes
            ]))
            server_result.add(MLMessage.GLOBAL_ROUND, self.args.round_idx)

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, server_result, client_indexes[receiver_id - 1]
                )

    def send_message_init_config(self, receive_id, server_result, client_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_CACHE_PATH, str(self.local_cache_path))
        # message.add_params(MyMessage.MSG_ARG_KEY_SERVER_RESULT, server_result)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, server_result)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, server_result, client_index
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        # message.add_params(MyMessage.MSG_ARG_KEY_SERVER_RESULT, server_result)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, server_result)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
