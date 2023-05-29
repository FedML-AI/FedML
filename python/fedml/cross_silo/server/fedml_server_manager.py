import json
import logging
import time

from fedml import mlops

from ...core import Context
from ...core.distributed.communication.message import Message
from ...core.distributed.fedml_comm_manager import FedMLCommManager
from ...core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from ...ml.selector.fedml_client_selector import FedMLClientSelector
from .message_define import MyMessage


class FedMLServerManager(FedMLCommManager):
    ONLINE_STATUS_FLAG = "ONLINE"
    RUN_FINISHED_STATUS_FLAG = "FINISHED"

    def __init__(
            self, args, aggregator, comm=None, rank=0, client_num=0, backend="MQTT_S3",
    ):
        super().__init__(args, comm, rank, client_num, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0

        self.client_online_status_mapping = {}
        self.client_real_ids = json.loads(args.client_id_list)

        self.client_finished_mapping = {}

        self.is_initialized = False
        self.client_id_list_in_this_round = None
        self.data_silo_index_list = None

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        logging.info("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.handle_message_client_status_update,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client,
        )

    def start_client_selection(self):
        """
        for high reliable training: asynchronous client selector based on client's network status
        FedMLClientSelector is a threading class which selects and waits for reliable clients and then callback
        """
        client_selector = FedMLClientSelector(
            name='FedMLClientSelector',
            round_idx=self.args.round_idx,
            client_num_in_total=self.args.client_num_per_round,
            client_num_per_round=self.args.client_num_per_round,
            client_online_status_mapping=self.client_online_status_mapping,
            client_real_ids=self.client_real_ids,
            callback_on_success=self.callback_on_success,
            callback_on_check_client_status=self.callback_on_check_client_status,
            callback_on_exception=self.callback_on_exception
        )
        client_selector.start()

    def callback_on_success(self, client_id_list_in_this_round, data_silo_index_list):
        self.client_id_list_in_this_round, self.data_silo_index_list = client_id_list_in_this_round, data_silo_index_list
        Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND, self.client_id_list_in_this_round)
        if self.args.round_idx == 0:
            self.send_init_msg()
        else:
            self.start_new_round()

    def callback_on_check_client_status(self, client_real_id, datasilo_index):
        self.send_message_check_client_status(client_real_id, datasilo_index)
    
    def callback_on_exception(self):
        raise Exception(f"failed to find enough reliable clients to join round {self.args.round_idx}")
        
    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()

        global_model_url = None
        global_model_key = None

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            global_model_url, global_model_key = self.send_message_init_config(
                client_id, global_model_params, self.data_silo_index_list[client_idx_in_this_round],
                global_model_url, global_model_key
            )
            client_idx_in_this_round += 1

        mlops.event("server.wait", event_started=True, event_value=str(self.args.round_idx))

        try:
            # get input type and shape for inference
            dummy_input_tensor = self.aggregator.get_dummy_input_tensor()

            model_net_url = mlops.log_training_model_net_info(
                self.aggregator.aggregator.model, dummy_input_tensor)

            # type and shape for later configuration
            input_shape, input_type = self.aggregator.get_input_shape_type()

            # Send output input size and type (saved as json) to s3,
            # and transfer when click "Create Model Card"
            model_input_url = mlops.log_training_model_input_info(
                list(input_shape), list(input_type))
        except Exception as e:
            logging.info("Cannot get dummy input size or shape for model serving")

        self.is_initialized = True
        mlops.log_aggregation_status(MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING)

    def start_new_round(self):
        global_model_params = self.aggregator.get_global_model_params()
        client_idx_in_this_round = 0
        global_model_url = None
        global_model_key = None
        for receiver_id in self.client_id_list_in_this_round:
            client_index = self.data_silo_index_list[client_idx_in_this_round]
            if type(global_model_params) is dict:
                # compatible with the old version that, user did not give {-1 : global_parms_dict}
                global_model_url, global_model_key = self.send_message_diff_sync_model_to_client(
                    receiver_id, global_model_params[client_index], client_index
                )
            else:
                global_model_url, global_model_key = self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_index, global_model_url, global_model_key
                )
            client_idx_in_this_round += 1

        # if user give {-1 : global_parms_dict}, then record global_model url separately
        if type(global_model_params) is dict and (-1 in global_model_params.keys()):
            global_model_url, global_model_key = self.send_message_diff_sync_model_to_client(
                -1, global_model_params[-1], -1
            )
            
        mlops.log_aggregated_model_info(
            self.args.round_idx, model_url=global_model_url,
        )

    def handle_message_connection_ready(self, msg_params):
        if not self.is_initialized:
            mlops.log_round_info(self.round_num, -1)
            # start an independent thread to select the clients asynchronously
            self.start_client_selection()

    def process_online_status(self, client_status, msg_params):
        self.client_online_status_mapping[str(msg_params.get_sender_id())] = True

    def process_finished_status(self, client_status, msg_params):  # todo: for async
        self.client_finished_mapping[str(msg_params.get_sender_id())] = True

        all_client_is_finished = True
        for client_id in self.client_id_list_in_this_round:
            if not self.client_finished_mapping.get(str(client_id), False):
                all_client_is_finished = False
                break

        logging.info(
            "sender_id = %d, all_client_is_finished = %s" % (
                msg_params.get_sender_id(), str(all_client_is_finished))
        )

        if all_client_is_finished:
            mlops.log_aggregation_finished_status()
            time.sleep(5)
            self.finish()

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        logging.info(f"received client status {client_status}")
        if client_status == FedMLServerManager.ONLINE_STATUS_FLAG:
            self.process_online_status(client_status, msg_params)
        elif client_status == FedMLServerManager.RUN_FINISHED_STATUS_FLAG:
            self.process_finished_status(client_status, msg_params)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        mlops.event("comm_c2s", event_started=False, event_value=str(
            self.args.round_idx), event_edge_id=sender_id)

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        local_model_round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_INDEX)
        logging.info(f"===========local_model_round_idx = {local_model_round_idx}, current index = {self.args.round_idx}")

        current_global_step_on_server = int(self.args.round_idx)
        current_global_step_on_client = int(local_model_round_idx)
        if self.aggregator.whether_to_accept(current_global_step_on_server, current_global_step_on_client):
            self.aggregator.add_local_trained_result(
                current_global_step_on_server, current_global_step_on_client,
                self.client_real_ids.index(sender_id), model_params, local_sample_number
            )

            if self.aggregator.whether_to_aggregate():
                logging.info("==========start to aggregate============")
                mlops.event("server.wait", event_started=False,
                            event_value=str(self.args.round_idx))
                mlops.event("server.agg_and_eval", event_started=True,
                            event_value=str(self.args.round_idx))

                tick = time.time()
                self.global_model_params, model_list, model_list_idxes = self.aggregator.aggregate()

                # used by security-enabled setting (e.g., outlier removal algorithm)
                logging.info("self.client_id_list_in_this_round = {}".format(
                    self.client_id_list_in_this_round))
                new_client_id_list_in_this_round = []
                for client_idx in model_list_idxes:
                    new_client_id_list_in_this_round.append(
                        self.client_id_list_in_this_round[client_idx])
                logging.info("new_client_id_list_in_this_round = {}".format(new_client_id_list_in_this_round))
                Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND, new_client_id_list_in_this_round)

                MLOpsProfilerEvent.log_to_wandb(
                    {"AggregationTime": time.time() - tick, "round": self.args.round_idx})

                self.aggregator.test_on_server_for_all_clients(self.args.round_idx)
                self.aggregator.assess_contribution()

                mlops.event("server.agg_and_eval", event_started=False, event_value=str(self.args.round_idx))

                # send round info to the MQTT backend
                mlops.log_round_info(self.round_num, self.args.round_idx)

                if self.args.round_idx == 0:
                    MLOpsProfilerEvent.log_to_wandb({"BenchmarkStart": time.time()})

                logging.info(f"\n\n==========end {self.args.round_idx}-th/{self.round_num} round training===========\n")
                
                self.args.round_idx += 1
                if self.args.round_idx < self.round_num:
                    mlops.event("server.wait", event_started=True,
                                event_value=str(self.args.round_idx))
                    # start a new round
                    self.start_client_selection()

    def cleanup(self):
        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_finish(
                client_id, self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index,
                                 global_model_url=None, global_model_key=None):
        tick = time.time()
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        if global_model_url is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
        if global_model_key is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_INDEX, self.args.round_idx)
        self.send_message(message)
        global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)
        MLOpsProfilerEvent.log_to_wandb({"Communiaction/Send_Total": time.time() - tick})
        return global_model_url, global_model_key

    def send_message_check_client_status(self, receive_id, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS,
                          self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)

    def send_message_finish(self, receive_id, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)
        logging.info(
            "finish from send id {} to receive id {}.".format(message.get_sender_id(), message.get_receiver_id()))
        logging.info(" ====================send cleanup message to {}====================".format(
            str(datasilo_index)))

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index,
                                          global_model_url=None, global_model_key=None):
        tick = time.time()
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                          self.get_sender_id(), receive_id, )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)

        if global_model_url is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
        if global_model_key is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_INDEX, str(self.args.round_idx))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        MLOpsProfilerEvent.log_to_wandb({"Communiaction/Send_Total": time.time() - tick})

        global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)

        return global_model_url, global_model_key

    def send_message_diff_sync_model_to_client(self, receive_id, client_model_params, client_index):
        tick = time.time()
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, client_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        MLOpsProfilerEvent.log_to_wandb({"Communiaction/Send_Total": time.time() - tick})

        global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)

        return global_model_url, global_model_key
