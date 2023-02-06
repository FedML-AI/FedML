import json
import logging
import platform

import numpy as np

from fedml import mlops
from .lsa_message_define import MyMessage
from ...core.distributed.fedml_comm_manager import FedMLCommManager
from ...core.distributed.communication.message import Message
from ...core.mpc.lightsecagg import (
    compute_aggregate_encoded_mask,
    mask_encoding,
    model_masking,
    model_dimension,
    transform_tensor_to_finite,
)


class FedMLClientManager(FedMLCommManager):
    def __init__(self, args, trainer, comm=None, client_rank=0, client_num=0, backend="MPI"):
        super().__init__(args, comm, client_rank, client_num, backend)
        self.args = args
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

        self.local_mask = None
        self.encoded_mask_dict = dict()
        self.flag_encoded_mask_dict = dict()
        self.worker_num = client_num - 1
        self.dimensions = []
        self.total_dimension = None
        for idx in range(self.worker_num):
            self.flag_encoded_mask_dict[idx] = False

        # new added parameters in main file
        # self.targeted_number_active_clients = args.targeted_number_active_clients
        # self.privacy_guarantee = args.privacy_guarantee
        self.targeted_number_active_clients = args.worker_num
        self.privacy_guarantee = int(np.floor(args.worker_num / 2))
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter

        self.client_real_ids = json.loads(args.client_id_list)
        logging.info("self.client_real_ids = {}".format(self.client_real_ids))
        # for the client, len(self.client_real_ids)==1: we only specify its client id in the list, not including others.
        self.client_real_id = self.client_real_ids[0]

        self.has_sent_online_msg = False
        self.sys_stats_process = None

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.handle_message_check_status
        )

        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_ENCODED_MASK_TO_CLIENT, self.handle_message_receive_encoded_mask_from_server,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT, self.handle_message_receive_active_from_server,
        )

    def handle_message_connection_ready(self, msg_params):
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(0)

            mlops.log_sys_perf(self.args)

    def handle_message_check_status(self, msg_params):
        self.send_client_status(0)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        logging.info("client_index = %s" % str(client_index))

        # Notify MLOps with training status.
        self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING)

        self.dimensions, self.total_dimension = model_dimension(global_model_params)

        self.trainer.update_dataset(int(client_index))
        self.trainer.update_model(global_model_params)
        self.round_idx = 0
        self.__offline()

    def handle_message_receive_encoded_mask_from_server(self, msg_params):
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)
        client_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        # logging.info(
        #     "Client %d receive encoded_mask = %s from Client %d"
        #     % (self.get_sender_id(), encoded_mask, client_id)
        # )
        self.add_encoded_mask(client_id - 1, encoded_mask)
        b_all_received = self.check_whether_all_encoded_mask_receive()
        if b_all_received:
            # TODO: performance optimization:
            #  local training can overlap the mask encoding and exchange step
            self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_dataset(int(client_index))
        self.trainer.update_model(model_params)

        if self.round_idx == self.num_rounds - 1:
            mlops.log_training_finished_status()

            self.finish()
            return
        self.round_idx += 1
        self.__offline()

    def handle_message_receive_active_from_server(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # Receive the set of active client id in first round
        active_clients_first_round = msg_params.get(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS)
        logging.info(
            "Client %d receive active_clients in the first round = %s"
            % (self.get_sender_id(), active_clients_first_round)
        )

        # Compute the aggregate of encoded masks for the active clients
        p = self.prime_number
        aggregate_encoded_mask = compute_aggregate_encoded_mask(self.encoded_mask_dict, p, active_clients_first_round)

        # Send the aggregate of encoded mask to server
        self.send_aggregate_encoded_mask_to_server(0, aggregate_encoded_mask)

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def send_client_status(self, receive_id, status="ONLINE"):
        logging.info("send_client_status")
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
        mlops.log_training_status(status)

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        mlops.event("comm_c2s", event_started=True, event_value=str(self.round_idx))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id,)

        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

        mlops.log_client_model_info(
            self.round_idx + 1, self.num_rounds, model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def send_encoded_mask_to_server(self, receive_id, encoded_mask):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_ENCODED_MASK, encoded_mask)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, receive_id)
        self.send_message(message)

    def send_aggregate_encoded_mask_to_server(self, receive_id, aggregate_encoded_mask):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MASK_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATE_ENCODED_MASK, aggregate_encoded_mask)
        self.send_message(message)

    def add_encoded_mask(self, index, encoded_mask):
        self.encoded_mask_dict[index] = encoded_mask
        self.flag_encoded_mask_dict[index] = True

    def check_whether_all_encoded_mask_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_encoded_mask_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_encoded_mask_dict[idx] = False
        return True

    def encoded_mask_sharing(self, encoded_mask_set):
        for receive_id in range(1, self.size + 1):
            print(receive_id)
            print("the size is ", self.size)
            encoded_mask = encoded_mask_set[receive_id - 1]
            if receive_id != self.get_sender_id():
                encoded_mask = encoded_mask.tolist()
                self.send_encoded_mask_to_server(receive_id, encoded_mask)
            else:
                self.encoded_mask_dict[receive_id - 1] = encoded_mask
                self.flag_encoded_mask_dict[receive_id - 1] = True

    def __offline(self):
        # Encoding the local generated mask
        logging.info("#######Client %d offline encoding round_id = %d######" % (self.get_sender_id(), self.round_idx))

        # encoded_mask_set = self.mask_encoding()
        d = self.total_dimension
        N = self.size
        U = self.targeted_number_active_clients
        T = self.privacy_guarantee
        p = self.prime_number
        logging.info("d = {}, N = {}, U = {}, T = {}, p = {}".format(d, N, U, T, p))
        d = int(np.ceil(float(d) / (U - T))) * (U - T)
        # For debugging
        self.local_mask = np.random.randint(p, size=(d, 1))
        # logging.info("local mask = {}".format(self.local_mask))
        # self.local_mask = np.zeros((d, 1)).astype("int64")
        print("new d is ", d)
        encoded_mask_set = mask_encoding(d, N, U, T, p, self.local_mask)

        # Send the encoded masks to other clients (via server)
        logging.info("begin share")
        self.encoded_mask_sharing(encoded_mask_set)
        logging.info("finish share")

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        mlops.event("train", event_started=True, event_value=str(self.round_idx))

        weights, local_sample_num = self.trainer.train(self.round_idx)
        # logging.info(
        #     "Client %d original weights = %s" % (self.get_sender_id(), weights)
        # )

        mlops.event("train", event_started=False, event_value=str(self.round_idx))

        # Convert the model from real to finite
        p = self.prime_number
        q_bits = self.precision_parameter
        weights_finite = transform_tensor_to_finite(weights, p, q_bits)

        # Mask the local model
        masked_weights = model_masking(weights_finite, self.dimensions, self.local_mask, self.prime_number)
        # logging.info(
        #     "Client %d send encode weights = %s"
        #     % (self.get_sender_id(), masked_weights)
        # )

        self.send_model_to_server(0, masked_weights, local_sample_num)

    def run(self):
        super().run()
