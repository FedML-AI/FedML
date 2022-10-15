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
        client_schedule = self.aggregator.generate_client_schedule(self.args.round_idx, client_indexes)
        self.training_num_in_round = self.aggregator.get_training_num_in_round(client_indexes)
        # global_model_params = self.aggregator.get_global_model_params()
        server_result = self.aggregator.get_init_server_result()
        server_result[MLMessage.TRAINING_NUM_IN_ROUND] = self.training_num_in_round
        server_result[MLMessage.SAMPLE_NUM_DICT] = dict([
            (client_index, self.aggregator.train_data_local_num_dict[client_index]) for client_index in client_indexes
        ])
        logging.info(f"client_indexes = {client_indexes}")

        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, server_result, client_schedule
            )

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # client_result = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_RESULT)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        # client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        local_agg_client_result = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_RESULT)
        local_sample_num_dict = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        client_runtime_info = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO)

        if hasattr(self.args, "tracking_runtime") and self.args.tracking_runtime and self.args.enable_wandb:
            runtime_to_wandb = {}
            for client_id, runtime in client_runtime_info.items():
                runtime_to_wandb[f"Runtime_w{sender_id - 1}_c{client_id}_n{local_sample_num_dict[client_id]}"] = runtime
            wandb.log(runtime_to_wandb)

        self.aggregator.record_client_runtime(sender_id - 1, client_runtime_info)
        assert hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq
        self.aggregator.global_aggregate_seq(sender_id - 1, local_agg_client_result, local_sample_num_dict)
        # if hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq:
        #     self.aggregator.global_aggregate_seq(worker_index, local_agg_client_result, local_sample_num_dict, training_num_in_round)
        # else:
        #     self.aggregator.add_local_trained_result(sender_id - 1, client_result, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.enable_wandb:
                wandb.log({"RunTimeOneRound": time.time() - self.previous_time, "round": self.args.round_idx})
                # things_to_wandb["RunTimeOneRound"] = time.time() - self.previous_time
                self.previous_time = time.time()
            server_result = self.aggregator.aggregate()
            current_time = time.time()
            self.aggregator.test_on_server_for_all_clients(self.args.round_idx)
            if self.args.enable_wandb:
                wandb.log({"TestTimeOneRound": time.time() - current_time, "round": self.args.round_idx})
                # things_to_wandb["TestTimeOneRound"] = time.time() - self.previous_time

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
            client_schedule = self.aggregator.generate_client_schedule(self.args.round_idx, client_indexes)
            self.training_num_in_round = self.aggregator.get_training_num_in_round(client_indexes)
            # global_model_params = self.aggregator.get_global_model_params()

            server_result[MLMessage.TRAINING_NUM_IN_ROUND] = self.training_num_in_round
            server_result[MLMessage.SAMPLE_NUM_DICT] = dict([
                    (client_index, self.aggregator.train_data_local_num_dict[client_index]) for client_index in client_indexes
                ])

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, server_result, client_schedule
                )

    def send_message_init_config(self, receive_id, server_result, client_schedule):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        # message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_CACHE_PATH, str(self.local_cache_path))
        message.add_params(MyMessage.MSG_ARG_KEY_SERVER_RESULT, server_result)
        # message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE, client_schedule)
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, server_result, client_schedule
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        # message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_SERVER_RESULT, server_result)
        # message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE, client_schedule)
        self.send_message(message)
