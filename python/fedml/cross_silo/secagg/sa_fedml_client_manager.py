import json
import logging
import platform
import copy
import numpy as np

from fedml import mlops
from .sa_message_define import MyMessage
from ...core.distributed.fedml_comm_manager import FedMLCommManager
from ...core.distributed.communication.message import Message
from ...core.mpc.secagg import (
    my_pk_gen,
    BGW_encoding,
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

        self.worker_num = client_num
        self.dimensions = []
        self.total_dimension = None

        # for secagg
        self.num_pk_per_user = 2
        self.targeted_number_active_clients = args.worker_num
        self.privacy_guarantee = int(np.floor(args.worker_num / 2))
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter
        self.public_key_others = np.empty(self.num_pk_per_user * self.worker_num).astype("int64")
        self.b_u_SS_others = np.empty((self.worker_num, self.worker_num), dtype="int64")
        self.s_sk_SS_others = np.empty((self.worker_num, self.worker_num), dtype="int64")

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
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_OTHER_PK_TO_CLIENT, self.handle_message_receive_pk_others,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_OTHER_SS_TO_CLIENT, self.handle_message_receive_ss_others,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_ACTIVE_CLIENT_LIST, self.handle_message_receive_active_from_server,
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
        if (not self.dimensions) or (not self.total_dimension):
            self.dimensions, self.total_dimension = model_dimension(model_params)
        self.__offline()

    def handle_message_receive_pk_others(self, msg_params):
        self.public_key_others = msg_params.get(MyMessage.MSG_ARG_KEY_PK_OTHERS)
        logging.info(" self.public_key_others = {}".format( self.public_key_others))
        self.public_key_others = np.reshape(self.public_key_others, (self.num_pk_per_user, self.worker_num))

    def handle_message_receive_ss_others(self, msg_params):
        self.s_sk_SS_others = msg_params.get(MyMessage.MSG_ARG_KEY_SK_SS_OTHERS).flatten()
        self.b_u_SS_others = msg_params.get(MyMessage.MSG_ARG_KEY_B_SS_OTHERS).flatten()
        self.s_pk_list = self.public_key_others[1, :]
        self.s_uv = np.mod(self.s_pk_list * self.my_s_sk, self.prime_number)
        self.__train()

    def handle_message_receive_active_from_server(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # Receive the set of active client id in first round
        active_clients = msg_params.get(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS)
        # 3.1. Send SS
        active_clients_dict = dict()
        for client in active_clients:
            active_clients_dict[client] = True
        SS_info = np.empty(self.worker_num, dtype="int64")
        for i in range(self.worker_num):
            if i in active_clients_dict:
                SS_info[i] = self.b_u_SS_others[i]
            else:
                SS_info[i] = self.s_sk_SS_others[i]
        self._send_others_ss_to_server(SS_info)

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
            self.round_idx + 1, self.num_rounds,  model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def _send_public_key_to_sever(self, public_key):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_PK_TO_SERVER, self.get_sender_id(), 0
        )
        message.add_params(MyMessage.MSG_ARG_KEY_PK, public_key)
        self.send_message(message)

    def _send_secret_share_to_sever(self, b_u_SS, s_sk_SS):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_SS_TO_SERVER, self.get_sender_id(), 0
        )
        message.add_params(MyMessage.MSG_ARG_KEY_B_SS, b_u_SS)
        message.add_params(MyMessage.MSG_ARG_KEY_SK_SS, s_sk_SS)
        self.send_message(message)

    def _send_others_ss_to_server(self, ss_info):

        # for j, k in enumerate(self.finite_w):
            # if j == 0:
            #     logging.info("Sent from %d" % (self.rank - 1))
            #     logging.info(self.finite_w[k][0])
            #     break

        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_SS_OTHERS_TO_SERVER,
            self.get_sender_id(),
            0,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_FINITE_W, self.finite_w)
        message.add_params(MyMessage.MSG_ARG_KEY_INFINITE_W, self.infinite_w)
        message.add_params(MyMessage.MSG_ARG_KEY_MASK, self.local_mask)
        message.add_params(MyMessage.MSG_ARG_KEY_SS_OTHERS, ss_info)
        self.send_message(message)

    def get_model_dimension(self, weights):
        self.dimensions, self.total_dimension = model_dimension(weights)

    def mask(self, weights):
        if (not self.dimensions) or (not self.total_dimension):
            self.dimensions, self.total_dimension = self.get_model_dimension(weights)
        q_bits = self.precision_parameter

        self.infinite_w = copy.deepcopy(weights)

        weights_finite = transform_tensor_to_finite(weights, self.prime_number, q_bits)

        self.finite_w = copy.deepcopy(weights_finite)

        d = self.total_dimension
        self.local_mask = np.zeros(d, dtype="int")
        for i in range(1, self.worker_num + 1):
            if self.rank == i:
                np.random.seed(self.b_u)
                temp = np.random.randint(0, self.prime_number, size=d).astype(int)
                logging.info("b for %d to %d" % (self.rank, i))
                logging.info(temp)
                self.local_mask = np.mod(self.local_mask + temp, self.prime_number)
                # temp = np.zeros(d,dtype='int')
            elif self.rank > i:
                np.random.seed(self.s_uv[i - 1])
                ##################################
                # Debugging Block Start #
                logging.info("*****************")
                logging.info(self.s_uv[i - 1])
                logging.info("{},{}".format(self.rank - 1, i - 1))
                # Debugging Block End #
                ##################################
                temp = np.random.randint(0, self.prime_number, size=d).astype(int)
                logging.info("s for %d to %d" % (self.rank, i))
                logging.info(temp)
                # if self.rank == 1:
                #    print '############ (seed, temp)=', self.s_uv[i-1], temp
                self.local_mask = np.mod(self.local_mask + temp, self.prime_number)
            else:
                np.random.seed(self.s_uv[i - 1])
                ##################################
                # Debugging Block Start #
                logging.info("*****************")
                logging.info(self.s_uv[i - 1])
                logging.info("{},{}".format(self.rank - 1, i - 1))
                # Debugging Block End #
                ##################################
                temp = -np.random.randint(0, self.prime_number, size=d).astype(int)
                logging.info("s for %d to %d" % (self.rank, i))
                logging.info(temp)
                # if self.rank == 1:
                #    print '############ (seed, temp)=', self.s_uv[i-1], temp
                self.local_mask = np.mod(self.local_mask + temp, self.prime_number)
        logging.info("Client")
        logging.info(self.rank)
        masked_weights = model_masking(weights_finite, self.dimensions, self.local_mask, self.prime_number)

        return masked_weights

    def __offline(self):
        np.random.seed(self.rank)
        self.sk = np.random.randint(0, self.prime_number, size=(2)).astype("int64")
        self.pk = my_pk_gen(self.sk, self.prime_number, 0)
        self.key = np.concatenate((self.pk, self.sk))  # length=4 : c_pk, s_pk, c_sk, s_sk

        self._send_public_key_to_sever(self.key[0:2])

        self.my_s_sk = self.key[3]
        self.my_c_sk = self.key[2]

        self.b_u = self.my_c_sk

        self.SS_input = np.reshape(np.array([self.my_c_sk, self.my_s_sk]), (2, 1))
        self.my_SS = BGW_encoding(self.SS_input, self.worker_num, self.privacy_guarantee, self.prime_number)

        self.b_u_SS = self.my_SS[:, 0, 0].astype("int64")
        self.s_sk_SS = self.my_SS[:, 1, 0].astype("int64")
        logging.info("seed b_u for use in %d", self.get_sender_id() - 1)
        logging.info(self.b_u)
        logging.info(self.b_u_SS)
        self._send_secret_share_to_sever(self.b_u_SS, self.s_sk_SS)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        mlops.event("train", event_started=True, event_value=str(self.round_idx))

        weights, local_sample_num = self.trainer.train(self.round_idx)
        # logging.info(
        #     "Client %d original weights = %s" % (self.get_sender_id(), weights)
        # )
        mlops.event("train", event_started=False, event_value=str(self.round_idx))

        # Mask the local model
        masked_weights = self.mask(weights)
        # logging.info(
        #     "Client %d send encode weights = %s"
        #     % (self.get_sender_id(), masked_weights)
        # )

        self.send_model_to_server(0, masked_weights, local_sample_num)

    def run(self):
        super().run()
