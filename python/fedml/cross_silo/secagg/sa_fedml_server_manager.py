import json
import logging
import time
from time import sleep
import torch

import numpy as np

from fedml import mlops
from .sa_message_define import MyMessage
from ...core.distributed.fedml_comm_manager import FedMLCommManager
from ...core.distributed.communication.message import Message
from ...core.mpc.secagg import (
    BGW_decoding,
    transform_finite_to_tensor,
    model_dimension,
)


class FedMLServerManager(FedMLCommManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        client_rank=0,
        client_num=0,
        backend="MQTT_S3",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, client_rank, client_num, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

        self.client_online_mapping = {}
        self.client_real_ids = json.loads(args.client_id_list)

        ### new added parameters in main file ###
        # self.targeted_number_active_clients = args.targeted_number_active_clients
        # self.privacy_guarantee = args.privacy_guarantee
        self.targeted_number_active_clients = args.worker_num
        self.privacy_guarantee = int(np.floor(args.worker_num / 2))
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter
        
        self.worker_num = args.worker_num
        self.num_pk_per_user = 2
        self.public_keys_received = dict()
        self.ss_received = dict()
        self.public_key_list = dict()
        self.b_u_SS_list = dict()
        self.s_sk_SS_list = dict()
        self.SS_rx = dict()
        self.model_dict = dict()
        self.client_aggregation_id = dict()
        self.participating_clients = dict()
        self.participating_clients_num = dict()
        self.first_round_clients = dict()
        self.second_round_clients = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.sample_num_dict = dict()

        ##################################
        # Debugging Block Start #
        self.finite_w = dict()
        self.infinite_w = dict()
        self.local_masks = dict()
        # Debugging Block End #
        ##################################

        self.aggregated_model_url = None

        self.is_initialized = False
        self.client_id_list_in_this_round = None
        self.data_silo_index_list = None

    def run(self):
        super().run()

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_init_config(
                client_id,
                global_model_params,
                self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1

        mlops.event("server.wait", event_started=True, event_value=str(self.round_idx))

    def register_message_receive_handlers(self):
        print("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_messag_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS,
            self.handle_message_client_status_update,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )
        
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_SS_TO_SERVER,
            self._handle_message_receive_ss,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_PK_TO_SERVER,
            self._handle_message_receive_public_key,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_SS_OTHERS_TO_SERVER,
            self._handle_message_receive_ss_others_from_client,
        )

    def handle_messag_connection_ready(self, msg_params):
        self.client_id_list_in_this_round = self.aggregator.client_selection(
            self.round_idx, self.client_real_ids, self.args.client_num_per_round
        )
        self.data_silo_index_list = self.aggregator.data_silo_selection(
            self.round_idx,
            self.args.client_num_in_total,
            len(self.client_id_list_in_this_round),
        )
        if not self.is_initialized:
            mlops.log_round_info(self.round_num, -1)

            # check client status in case that some clients start earlier than the server
            client_idx_in_this_round = 0
            for client_id in self.client_id_list_in_this_round:
                self.send_message_check_client_status(
                    client_id, self.data_silo_index_list[client_idx_in_this_round],
                )
                client_idx_in_this_round += 1

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        if client_status == "ONLINE":
            self.client_online_mapping[str(msg_params.get_sender_id())] = True

        mlops.log_aggregation_status(MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING)

        all_client_is_online = True
        for client_id in self.client_id_list_in_this_round:
            if not self.client_online_mapping.get(str(client_id), False):
                all_client_is_online = False
                break

        logging.info(
            "sender_id = %d, all_client_is_online = %s"
            % (msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            # send initialization message to all clients to start training
            self.send_init_msg()
            self.is_initialized = True

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        mlops.event(
            "comm_c2s",
            event_started=False,
            event_value=str(self.round_idx),
            event_edge_id=sender_id,
        )

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        self.active_clients_first_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        # TODO: add a timeout procedure here
        if b_all_received:
            # Specify the active clients for the first round and inform them
            for receiver_id in range(1, self.size + 1):
                self.send_message_to_active_client(
                    receiver_id, self.active_clients_first_round
                )
                
    def _handle_message_receive_public_key(self, msg_params):
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        public_key = msg_params.get(MyMessage.MSG_ARG_KEY_PK)
        local_aggregation_id = self._get_client_aggregation_id(sender_id - 1)
        self.public_key_list[local_aggregation_id][:, sender_id - 1] = public_key
        self.public_keys_received[local_aggregation_id] += 1
        if self._check_all_public_keys_received(local_aggregation_id):
            data = np.reshape(self.public_key_list[local_aggregation_id], self.num_pk_per_user * self.worker_num)
            for i in range(self.worker_num):
                self._send_public_key_others_to_user(i + 1, data)

    def _handle_message_receive_ss(self, msg_params):
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        b_u_SS = msg_params.get(MyMessage.MSG_ARG_KEY_B_SS)
        s_sk_SS = msg_params.get(MyMessage.MSG_ARG_KEY_SK_SS)
        local_aggregation_id = self._get_client_aggregation_id(sender_id - 1)
        self.b_u_SS_list[local_aggregation_id][sender_id - 1, :] = b_u_SS
        self.s_sk_SS_list[local_aggregation_id][sender_id - 1, :] = s_sk_SS
        self.ss_received[local_aggregation_id] += 1
        if self._check_all_ss_received(local_aggregation_id):
            for i in range(self.worker_num):
                self._send_ss_others_to_user(
                    i + 1, self.b_u_SS_list[local_aggregation_id][:,i], self.s_sk_SS_list[local_aggregation_id][:,i]
                )

    def unmask(self, server_model_params, active_clients, sample_num, unmasked_model_handler, aggregation_id=0):
        logging.info("@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(active_clients)
        self.model_dict[aggregation_id] = server_model_params
        self.sample_num_dict[aggregation_id] = sample_num
        self.first_round_clients[aggregation_id] = active_clients
        self.unmasked_model_handler = unmasked_model_handler
        for client in self.first_round_clients[aggregation_id]:
            self.flag_client_model_uploaded_dict[aggregation_id][client] = True

        for client_id in active_clients:
            self._send_message_to_active_client(client_id + 1, active_clients)

    def _handle_message_receive_ss_others_from_client(self, msg_params):
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        ss_others = msg_params.get(MyMessage.MSG_ARG_KEY_SS_OTHERS)
        local_aggregation_id = self._get_client_aggregation_id(sender_id - 1)
        b_all_received = self._check_ss_others_enough_received(local_aggregation_id)
        # TODO: Check if enough secret shares are collected
        # TODO: check for correctness
        if b_all_received:
            return

        ##################################
         # Debugging Block Start #
        self.finite_w[local_aggregation_id][sender_id - 1] = msg_params.get(MyMessage.MSG_ARG_KEY_FINITE_W)
        self.infinite_w[local_aggregation_id][sender_id - 1] = msg_params.get(MyMessage.MSG_ARG_KEY_INFINITE_W)
        self.local_masks[local_aggregation_id][sender_id - 1] = msg_params.get(MyMessage.MSG_ARG_KEY_MASK)
        for j,k in enumerate(self.finite_w[local_aggregation_id][sender_id - 1]):
            if j == 0:
                logging.info("Received from %d" % (sender_id - 1))
                logging.info(self.finite_w[local_aggregation_id][sender_id - 1][k][0])
                logging.info("SDFSDFSDFSDF")
                logging.info(self.model_dict[local_aggregation_id][sender_id - 1][k][0])
                break
        # assert 1 == 0
         # Debugging Block End #
        ##################################
        

        self.SS_rx[local_aggregation_id][:, sender_id - 1] = ss_others
        self.second_round_clients[local_aggregation_id].append(sender_id - 1)

        b_all_received = self._check_ss_others_enough_received(local_aggregation_id)
        # logging.info("Server: mask_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)

        # After receiving enough aggregate of encoded masks, server recovers the aggregate-model
        if b_all_received:
            active_clients_second_round = self.get_second_round_clients(local_aggregation_id)
            active_clients_first_round = self.get_first_round_clients(local_aggregation_id)
            sample_num = self.get_sample_num(local_aggregation_id)

            # Secure Model Aggregation
            global_model_params = self.aggregate_model_reconstruction(
                active_clients_first_round, active_clients_second_round, local_aggregation_id, sample_num
            )

            self.unmasked_model_handler(global_model_params, self.participating_clients[local_aggregation_id], local_aggregation_id)
            
    def aggregate_model_reconstruction(self, active_clients_first_round, active_clients_second_round, aggregation_id, sample_num):
        start_time = time.time()
        aggregate_mask = self.aggregate_mask_reconstruction(active_clients_second_round, aggregation_id)
        p = self.prime_number
        q_bits = self.precision_parameter
        logging.info("Server starts the reconstruction of aggregate_model")
        averaged_params = {}
        pos = 0



        ##################################
        # Debugging Block Start #
        for j, k in enumerate(self.model_dict[aggregation_id][active_clients_first_round[0]]):
            if j == 0:
                logging.info("1111111@###########@@@@@@@@@@@")
                logging.info(k)
                for l in active_clients_first_round:
                    logging.info(l)
                    logging.info(self.finite_w[aggregation_id][l][k][0])
                    logging.info(torch.from_numpy(np.array(self.finite_w[aggregation_id][l][k][0])))
        aggergated_finite_w = {}
        aggergated_infinite_w = {}
        # Debugging Block End #
        ##################################

        for j, k in enumerate(self.model_dict[aggregation_id][active_clients_first_round[0]]):
           ##################################
           # Debugging Block Start #
            aggergated_infinite_w[k] = 0
            aggergated_finite_w[k] = 0
           # Debugging Block End #
           ##################################
            averaged_params[k] = 0
            for i, client_idx in enumerate(active_clients_first_round):
                if not (client_idx in self.flag_client_model_uploaded_dict[aggregation_id]
                and self.flag_client_model_uploaded_dict[aggregation_id][client_idx]):
                    continue
                local_model_params = self.model_dict[aggregation_id][client_idx] 
                averaged_params[k] += local_model_params[k]
                ##################################
                # Debugging Block Start #
                aggergated_infinite_w[k] +=  torch.from_numpy(np.array(self.infinite_w[aggregation_id][client_idx][k]))
                aggergated_finite_w[k] +=  torch.from_numpy(np.array(self.finite_w[aggregation_id][client_idx][k]))
                if j == 0:
                    logging.info("2323232$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                    logging.info(client_idx)
                    logging.info(self.finite_w[aggregation_id][client_idx][k][0])
                aggergated_finite_w[k] = np.mod( aggergated_finite_w[k], p)
                    # Debugging Block End #
                    ##################################
                averaged_params[k] = np.mod( averaged_params[k], p)

            cur_shape = np.shape(averaged_params[k])
            d = self.dimensions[j]
            aggregate_mask = aggregate_mask.reshape((aggregate_mask.shape[0], 1))
            cur_mask = np.array(aggregate_mask[pos : pos + d, :])
            cur_mask = np.reshape(cur_mask, cur_shape)

            ##################################
            # Debugging Block Start #
            if j == 0:
                logging.info("@###########@@@@@@@@@@@")
                logging.info(k)
                mask_sum = 0
                local_masks = self.local_masks[aggregation_id]
                for l in active_clients_first_round:
                    cur_shape_1 = np.shape(averaged_params[k])
                    mask = local_masks[l].reshape((local_masks[l].shape[0], 1))
                    cur_mask_1 = np.array(mask[0 : 0 + d, :])
                    cur_mask_1 = np.reshape(cur_mask_1, cur_shape_1)
                    mask_sum += cur_mask_1
                    logging.info("^^^^^^^^^^^^^^^^^^^")
                    logging.info(l)
                    logging.info("Before Masking")
                    
                    logging.info(np.array(self.finite_w[aggregation_id][l][k]).shape)
                    logging.info(np.array(cur_mask_1.shape))
                    logging.info("Mask")
                    logging.info("Mask")
                    logging.info(cur_mask_1[0])
                    logging.info("After Masking")
                    logging.info(np.mod(torch.from_numpy(np.array(self.finite_w[aggregation_id][l][k][0])) + cur_mask_1[0] ,p))
                    logging.info("No Mod")
                    logging.info(torch.from_numpy(np.array(self.finite_w[aggregation_id][l][k][0])) + cur_mask_1[0])
                    logging.info(self.model_dict[aggregation_id][l][k][0])
                    logging.info("Diff")
                    logging.info((np.mod(torch.from_numpy(np.array(self.finite_w[aggregation_id][l][k][0])) + cur_mask_1[0] ,p) - self.model_dict[aggregation_id][l][k][0]))
                    logging.info(np.mod(torch.from_numpy(np.array(self.finite_w[aggregation_id][l][k][0])) + cur_mask_1[0] - self.model_dict[aggregation_id][l][k][0],p))
                    # assert((np.mod(torch.from_numpy(np.array(self.finite_w[aggregation_id][l][k])) + cur_mask_1 ,p) == self.model_dict[aggregation_id][l][k]).all())
                logging.info("Compare Masks")
                logging.info("Decoded")
                logging.info(np.mod(cur_mask[0],p))
                logging.info("Not coded")
                logging.info(np.mod(mask_sum[0],p))
                logging.info("Difference")
                logging.info(np.mod(aggergated_finite_w[k] + mask_sum -  averaged_params[k],p))
                # assert(np.mod(aggergated_finite_w[k] + mask_sum ,p) == np.mod(averaged_params[k],p)).all()
                # assert(np.mod( mask_sum,p) == np.mod(cur_mask,p)).all()


            # Debugging Block End #
            ##################################



            # Cancel out the aggregate-mask to recover the aggregate-model
            averaged_params[k] -= cur_mask
            averaged_params[k] = np.mod(averaged_params[k], p)
            pos += d


        ##################################
        # Debugging Block Start #
        logging.info("@#@#@#@###@#@#@##@@#@##")
        logging.info("Aggregation Difference")
        for j, k in enumerate(self.model_dict[aggregation_id][active_clients_first_round[0]]):
            if j < 5:
                logging.info(j)
                logging.info((averaged_params[k] - np.mod(aggergated_finite_w[k], p))[0])
        # Debugging Block End #
        ##################################


        # Convert the model from finite to real
        logging.info("Server converts the aggregate_model from finite to tensor")
        averaged_params = transform_finite_to_tensor(averaged_params, p, q_bits)

        ##################################
        # Debugging Block Start #
        logging.info("@#@#@#@###@#@#@##@@#@##")
        logging.info("Aggregation Difference Infinite")
        for j, k in enumerate(self.model_dict[aggregation_id][active_clients_first_round[0]]):
            if j < 5:
                logging.info(j)
                logging.info((averaged_params[k][0]))
                logging.info((aggergated_infinite_w[k][0]))
                logging.info((averaged_params[k] - aggergated_infinite_w[k])[0])
        # Debugging Block End #
        ##################################

        aggregated_sample_num = 0
        for client_idx in active_clients_first_round:
            aggregated_sample_num += sample_num[client_idx]

        for j, k in enumerate(self.model_dict[aggregation_id][active_clients_first_round[0]]):
            averaged_params[k] = averaged_params[k] / aggregated_sample_num

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def aggregate_mask_reconstruction(self, active_clients, aggregation_id):
        """
        Recover the aggregate-mask via decoding
        """
        d = self.total_dimension
        # N = self.worker_num
        # U = self.targeted_number_active_clients
        T = self.privacy_guarantee
        p = self.prime_number
        logging.debug("d = {}, T = {}, p = {}".format(d, T, p))

        aggregated_mask = 0
        
        for i in self.participating_clients[aggregation_id]:
            if (
                i in self.flag_client_model_uploaded_dict[aggregation_id]
                and self.flag_client_model_uploaded_dict[aggregation_id][i]
            ):
                SS_input = np.reshape(self.SS_rx[aggregation_id][i, active_clients[:T+1]], (T + 1, 1))
                b_u = BGW_decoding(SS_input, active_clients[:T+1], p)
                np.random.seed(b_u[0][0])
                mask = np.random.randint(0, p, size=d).astype(int)
                aggregated_mask += mask
                # z = np.mod(z - temp, p)
            else:
                mask = np.zeros(d, dtype="int")
                SS_input = np.reshape(self.SS_rx[aggregation_id][i, active_clients[:T+1]], (T + 1, 1))
                s_sk_dec = BGW_decoding(SS_input, active_clients[:T+1], p)
                for j in self.participating_clients[aggregation_id]:
                    s_pk_list_ = self.public_key_list[aggregation_id][1, :]
                    s_uv_dec = np.mod(s_sk_dec[0][0] * s_pk_list_[j], p)
                    logging.info("&&&&&&&&&&&&&&&&&&&&&&&")
                    logging.info(s_uv_dec)
                    logging.info("{},{}".format(i,j))
                    if j == i:
                        temp = np.zeros(d, dtype="int")
                    elif j < i:
                        np.random.seed(s_uv_dec)
                        temp = -np.random.randint(0, p, size=d).astype(int)
                    else:
                        # np.random.seed(s_uv[j-1])
                        np.random.seed(s_uv_dec)
                        temp = +np.random.randint(0, p, size=d).astype(int)
                    # print 'seed, temp=',s_uv_dec,temp
                    mask = np.mod(mask + temp, p)
                # print 'mask =', mask
                aggregated_mask += mask
            aggregated_mask = np.mod(aggregated_mask, p)


        return aggregated_mask
    # TODO: stop after receiving enough (T+1)
    # TODO: Add round finished to handle delayed resutls
    def _check_ss_others_enough_received(self, aggregation_id):
        if len(self.second_round_clients[aggregation_id]) == len(self.first_round_clients[aggregation_id]):
            return True
        return False

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

    def send_message_check_client_status(self, receive_id, datasilo_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)

    def send_message_finish(self, receive_id, datasilo_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)
        logging.info(
            " ====================send cleanup message to {}====================".format(
                str(datasilo_index)
            )
        )

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, client_index
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        mlops.log_aggregated_model_info(
            self.round_idx + 1,
            model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def send_message_to_active_client(self, receive_id, active_clients):
        logging.info(
            "Server send_message_to_active_client. receive_id = %d" % receive_id
        )
        message = Message(
            MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS, active_clients)
        self.send_message(message)
        
    def _send_public_key_others_to_user(self, receive_id, public_key_other):
        logging.info("Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SEND_PK_OTHERS_TO_CLIENT, self.server_manager.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_PK_OTHERS, public_key_other)
        self.server_manager.send_message(message)

    def _send_ss_others_to_user(self, receive_id, b_ss_others, sk_ss_others):
        logging.info("Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SEND_SS_OTHERS_TO_CLIENT, self.server_manager.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_B_SS_OTHERS, b_ss_others)
        message.add_params(MyMessage.MSG_ARG_KEY_SK_SS_OTHERS, sk_ss_others)
        self.server_manager.send_message(message)

    def _send_message_to_active_client(self, receive_id, active_clients):
        logging.info("Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT, self.server_manager.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS, active_clients)
        self.server_manager.send_message(message)