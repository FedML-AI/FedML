import json
import logging
import time
import traceback

from fedml import mlops

from .message_define import MyMessage
from .utils import write_tensor_dict_to_mnn
from ...core.distributed.communication.message import Message
from ...core.distributed.fedml_comm_manager import FedMLCommManager


class FedMLServerManager(FedMLCommManager):
    ONLINE_STATUS_FLAG = "ONLINE"
    RUN_FINISHED_STATUS_FLAG = "FINISHED"

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

        self.global_model_file_path = self.args.global_model_file_path
        self.model_file_cache_folder = self.args.model_file_cache_folder
        logging.info(
            "self.global_model_file_path = {}".format(self.global_model_file_path)
        )
        logging.info(
            "self.model_file_cache_folder = {}".format(self.model_file_cache_folder)
        )

        self.client_online_mapping = {}
        self.client_real_ids = json.loads(args.client_id_list)

        self.client_finished_mapping = {}

        self.is_initialized = False
        self.client_id_list_in_this_round = None
        self.data_silo_index_list = None

    def run(self):
        super().run()

    def start_train(self):
        start_train_json = {
            "edges": [
                {
                    "device_id": "62dcd04fa9bc672e",
                    "os_type": "Android",
                    "id": self.args.client_id_list,
                }
            ],
            "starttime": 1651635148113,
            "url": "http://fedml-server-agent-svc.fedml-aggregator-dev.svc.cluster.local:5001/api/start_run",
            "edgeids": [127],
            "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA1LCJhY2NvdW50IjoiYWxleC5saWFuZzIiLCJsb2dpblRpbWUiOiIxNjUxNjM0Njc0NDcwIiwiZXhwIjowfQ.miX2--XbaJab-sNPHzZcsMWcVOXPLQHFNXuK0oMAYiY",
            "urls": [],
            "userids": ["208"],
            "name": "hundred_daily",
            "runId": 189,
            "id": 169,
            "projectid": "169",
            "run_config": {
                "configName": "test-new-open",
                "userId": 208,
                "model_config": {},
                "packages_config": {
                    "server": "server-package.zip",
                    "linuxClient": "client-package.zip",
                    "serverUrl": "https://fedml.s3.us-west-1.amazonaws.com/1651440439347server-package.zip",
                    "linuxClientUrl": "https://fedml.s3.us-west-1.amazonaws.com/1651440442364client-package.zip",
                    "androidClient": "",
                    "androidClientUrl": "",
                    "androidClientVersion": "0",
                },
                "data_config": {
                    "privateLocalData": "",
                    "syntheticData": "",
                    "syntheticDataUrl": "",
                },
                "parameters": {
                    "model_args": {
                        "model_file_cache_folder": "./model_file_cache",
                        "model": "lr",
                        "global_model_file_path": "./model_file_cache/global_model.pt",
                    },
                    "device_args": {
                        "worker_num": 1,
                        "using_gpu": False,
                        "gpu_mapping_key": "mapping_default",
                        "gpu_mapping_file": "config/gpu_mapping.yaml",
                    },
                    "comm_args": {
                        "s3_config_path": "config/s3_config.yaml",
                        "backend": "MQTT_S3",
                        "mqtt_config_path": "config/mqtt_config.yaml",
                    },
                    "train_args": {
                        "batch_size": self.args.batch_size,
                        "weight_decay": self.args.weight_decay,
                        "client_num_per_round": self.args.client_num_per_round,
                        "client_num_in_total": self.args.client_num_in_total,
                        "comm_round": self.args.comm_round,
                        "client_optimizer": self.args.client_optimizer,
                        "client_id_list": self.args.client_id_list,
                        "epochs": self.args.epochs,
                        "learning_rate": self.args.learning_rate,
                        "federated_optimizer": self.args.federated_optimizer,
                    },
                    "environment_args": {"bootstrap": "config/bootstrap.sh"},
                    "validation_args": {"frequency_of_the_test": 1},
                    "common_args": {
                        "random_seed": 0,
                        "training_type": "cross_silo",
                        "using_mlops": False,
                    },
                    "data_args": {
                        "partition_method": self.args.partition_method,
                        "partition_alpha": self.args.partition_alpha,
                        "dataset": self.args.dataset,
                        "data_cache_dir": self.args.data_cache_dir,
                        "train_size": self.args.train_size,
                        "test_size": self.args.test_size,
                    },
                    "tracking_args": {
                        "wandb_project": "fedml",
                        "wandb_name": "fedml_torch_fedavg_mnist_lr",
                        "wandb_key": "ee0b5f53d949c84cee7decbe7a629e63fb2f8408",
                        "enable_wandb": False,
                        "log_file_dir": "./log",
                    },
                },
            },
            "timestamp": "1651635148138",
        }
        for client_id in self.client_real_ids:
            logging.info("com_manager_status - client_id = {}".format(client_id))
            self.send_message_json(
                "flserver_agent/" + str(client_id) + "/start_train",
                json.dumps(start_train_json),
            )

    def send_init_msg(self):
        """
        init - send model to client:
            MNN (file) which is from "model_file_path: config/lenet_mnist.mnn"
        C2S - received all models from clients:
            MNN (file) -> numpy -> pytorch -> aggregation -> numpy -> MNN (the same file)
        S2C - send the model to clients
            send MNN file
        """
        global_model_url = None
        global_model_key = None

        client_idx_in_this_round = 0
        for receiver_id in self.client_id_list_in_this_round:
            global_model_url, global_model_key = self.send_message_init_config(
                receiver_id,
                self.global_model_file_path,
                self.data_silo_index_list[client_idx_in_this_round],
                global_model_url, global_model_key
            )
            logging.info(f"global_model_url = {global_model_url}, global_model_key = {global_model_key}")
            client_idx_in_this_round += 1

        mlops.event("server.wait", event_started=True, event_value=str(self.args.round_idx))

        # Todo: for serving the cross-device model,
        #       how to transform it to pytorch and upload the model network to ModelOps
        # mlops.log_training_model_net_info(self.aggregator.aggregator.model)

    def register_message_receive_handlers(self):
        print("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS,
            self.handle_message_client_status_update,
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

    def process_online_status(self, client_status, msg_params):
        self.client_online_mapping[str(msg_params.get_sender_id())] = True

        logging.info("self.client_online_mapping = {}".format(self.client_online_mapping))

        all_client_is_online = True
        for client_id in self.client_id_list_in_this_round:
            if not self.client_online_mapping.get(str(client_id), False):
                all_client_is_online = False
                break

        logging.info(
            "sender_id = %d, all_client_is_online = %s" % (msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            mlops.log_aggregation_status(MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING)

            # send initialization message to all clients to start training
            self.send_init_msg()
            self.is_initialized = True

    def process_finished_status(self, client_status, msg_params):
        self.client_finished_mapping[str(msg_params.get_sender_id())] = True

        all_client_is_finished = True
        for client_id in self.client_id_list_in_this_round:
            if not self.client_finished_mapping.get(str(client_id), False):
                all_client_is_finished = False
                break

        logging.info(
            "sender_id = %d, all_client_is_finished = %s" % (msg_params.get_sender_id(), str(all_client_is_finished))
        )

        if all_client_is_finished:
            mlops.log_aggregation_finished_status()
            time.sleep(5)
            self.finish()

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        if client_status == FedMLServerManager.ONLINE_STATUS_FLAG:
            self.process_online_status(client_status, msg_params)
        elif client_status == FedMLServerManager.RUN_FINISHED_STATUS_FLAG:
            self.process_finished_status(client_status, msg_params)

    def handle_message_connection_ready(self, msg_params):
        if not self.is_initialized:
            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.args.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            self.data_silo_index_list = self.aggregator.data_silo_selection(
                self.args.round_idx,
                self.args.client_num_in_total,
                len(self.client_id_list_in_this_round),
            )
            logging.info(
                "client_id_list_in_this_round = {}, data_silo_index_list = {}".format(
                    self.client_id_list_in_this_round, self.data_silo_index_list
                )
            )

            mlops.log_round_info(self.round_num, -1)

            # check client status in case that some clients start earlier than the server
            client_idx_in_this_round = 0
            for client_id in self.client_id_list_in_this_round:
                try:
                    self.send_message_check_client_status(
                        client_id, self.data_silo_index_list[client_idx_in_this_round],
                    )
                    logging.info("Connection ready for client: " + str(client_id))
                except Exception as e:
                    logging.info("Connection not ready for client: {}".format(
                        str(client_id), traceback.format_exc()))
                client_idx_in_this_round += 1

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)

        mlops.event("comm_c2s", event_started=False, event_value=str(self.args.round_idx), event_edge_id=sender_id)

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = %s " % str(b_all_received))
        if b_all_received:
            logging.info("=================================================")
            logging.info(
                "=========== ROUND {} IS FINISHED!!! =============".format(
                    self.args.round_idx
                )
            )
            logging.info("=================================================")

            mlops.event("server.wait", event_started=False, event_value=str(self.args.round_idx))

            mlops.event("server.agg_and_eval", event_started=True, event_value=str(self.args.round_idx))

            global_model_params = self.aggregator.aggregate()
            
            # self.aggregator.test_on_server_for_all_clients(
            #     self.args.round_idx, self.global_model_file_path
            # )
            
            write_tensor_dict_to_mnn(self.global_model_file_path, global_model_params)
            self.aggregator.test_on_server_for_all_clients_mnn(
                self.global_model_file_path, self.args.round_idx
            )
            
            mlops.event("server.agg_and_eval", event_started=False, event_value=str(self.args.round_idx))


            # send round info to the MQTT backend
            mlops.log_round_info(self.round_num, self.args.round_idx)

            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.args.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            self.data_silo_index_list = self.aggregator.data_silo_selection(
                self.args.round_idx,
                self.args.client_num_in_total,
                len(self.client_id_list_in_this_round),
            )

            client_idx_in_this_round = 0
            global_model_url = None
            global_model_key = None
            logging.info("round idx {}, client_num_in_total {}, data_silo_index_list length {},"
                         "client_id_list_in_this_round length {}.".format(
                self.args.round_idx, self.args.client_num_in_total,
                len(self.data_silo_index_list), len(self.client_id_list_in_this_round)))
            for receiver_id in self.client_id_list_in_this_round:
                global_model_url, global_model_key = self.send_message_sync_model_to_client(
                    receiver_id,
                    self.global_model_file_path,
                    self.data_silo_index_list[client_idx_in_this_round],
                    global_model_url,
                    global_model_key
                )
                client_idx_in_this_round += 1

            self.args.round_idx += 1
            mlops.log_aggregated_model_info(
                self.args.round_idx, model_url=global_model_url,
            )

            logging.info("\n\n==========end {}-th round training===========\n".format(self.args.round_idx))
            if self.args.round_idx < self.round_num:
                mlops.event("server.wait", event_started=True, event_value=str(self.args.round_idx))

    def cleanup(self):
        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_finish(
                client_id, self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1

    def send_message_finish(self, receive_id, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)
        logging.info(
            "finish from send id {} to receive id {}.".format(message.get_sender_id(), message.get_receiver_id()))
        logging.info(" ====================send cleanup message to {}====================".format(str(datasilo_index)))

    def send_message_init_config(self, receive_id, global_model_params, client_index,
                                 global_model_url, global_model_key):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        if global_model_url is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
        if global_model_key is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
        logging.info("global_model_params = {}".format(global_model_params))
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "AndroidClient")
        self.send_message(message)

        global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)
        return global_model_url, global_model_key

    def send_message_check_client_status(self, receive_id, datasilo_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
            self, receive_id, global_model_params, data_silo_index,
            global_model_url=None, global_model_key=None
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        if global_model_url is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
        if global_model_key is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(data_silo_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "AndroidClient")
        self.send_message(message)

        global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)

        return global_model_url, global_model_key
