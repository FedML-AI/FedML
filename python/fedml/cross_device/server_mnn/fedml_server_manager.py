import json
import time

from .message_define import MyMessage
from .utils import write_tensor_dict_to_mnn
from ...core.distributed.communication.message import Message
from ...core.distributed.server.server_manager import ServerManager
from ...core.mlops import MLOpsMetrics, MLOpsProfilerEvent
import logging

from ...core.mlops.mlops_configs import MLOpsConfigs


class FedMLServerManager(ServerManager):
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
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

        self.client_stubs = {}
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

        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics = MLOpsMetrics()
            self.mlops_metrics.set_messenger(self.com_manager_status)
            self.mlops_event = MLOpsProfilerEvent(self.args)
            self.aggregator.set_mlops_metrics_logger(self.mlops_metrics)

        self.start_running_time = 0.0
        self.aggregated_model_url = None
        self.event_sdk = MLOpsProfilerEvent(self.args)

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
            self.com_manager_status.send_message_json(
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
        client_id_list_in_this_round = self.aggregator.client_selection(
            self.round_idx, self.client_real_ids, self.args.client_num_per_round
        )
        data_silo_index_list = self.aggregator.data_silo_selection(
            self.round_idx,
            self.args.client_num_in_total,
            len(client_id_list_in_this_round),
        )
        logging.info(
            "client_id_list_in_this_round = {}, data_silo_index_list = {}".format(
                client_id_list_in_this_round, data_silo_index_list
            )
        )

        client_idx_in_this_round = 0
        for receiver_id in client_id_list_in_this_round:
            self.send_message_init_config(
                receiver_id,
                self.global_model_file_path,
                data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1

        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_started(
                "server.wait", event_value=str(self.round_idx)
            )

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

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        if client_status == "ONLINE":
            self.client_online_mapping[str(msg_params.get_sender_id())] = True

        # notify MLOps with RUNNING status
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics.report_server_training_status(
                self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING
            )

        all_client_is_online = True
        for client_id in self.client_real_ids:
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

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_ended(
                "comm_c2s", event_value=str(self.round_idx), event_edge_id=sender_id
            )

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        logging.info("model_params = {}".format(model_params))

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = %s " % str(b_all_received))
        if b_all_received:
            if hasattr(self.args, "backend") and self.args.using_mlops:
                self.mlops_event.log_event_ended(
                    "server.wait", event_value=str(self.round_idx)
                )
                self.mlops_event.log_event_started(
                    "aggregate", event_value=str(self.round_idx)
                )

            global_model_params = self.aggregator.aggregate()
            write_tensor_dict_to_mnn(self.global_model_file_path, global_model_params)

            if hasattr(self.args, "backend") and self.args.using_mlops:
                self.mlops_event.log_event_ended(
                    "aggregate", event_value=str(self.round_idx)
                )

            self.aggregator.test_on_server_for_all_clients(
                self.global_model_file_path, self.round_idx
            )

            # send round info to the MQTT backend
            round_info = {
                "run_id": self.args.run_id,
                "round_index": self.round_idx,
                "total_rounds": self.round_num,
                "running_time": round(time.time() - self.start_running_time, 4),
            }
            if hasattr(self.args, "backend") and self.args.using_mlops:
                self.mlops_metrics.report_server_training_round_info(round_info)

            client_id_list_in_this_round = self.aggregator.client_selection(
                self.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            data_silo_index_list = self.aggregator.data_silo_selection(
                self.round_idx,
                self.args.client_num_in_total,
                len(client_id_list_in_this_round),
            )

            client_idx_in_this_round = 0
            for receiver_id in client_id_list_in_this_round:
                self.send_message_sync_model_to_client(
                    receiver_id,
                    self.global_model_file_path,
                    data_silo_index_list[client_idx_in_this_round],
                )
                client_idx_in_this_round += 1

            if hasattr(self.args, "backend") and self.args.using_mlops:
                model_info = {
                    "run_id": self.args.run_id,
                    "round_idx": self.round_idx + 1,
                    "global_aggregated_model_s3_address": self.aggregated_model_url,
                }
                self.mlops_metrics.report_aggregated_model_info(model_info)
                self.aggregated_model_url = None

            self.round_idx += 1
            if self.round_idx == self.round_num:
                if hasattr(self.args, "backend") and self.args.using_mlops:
                    self.mlops_metrics.report_server_id_status(
                        self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_FINISHED
                    )
                self.finish()
                return
            else:
                if hasattr(self.args, "backend") and self.args.using_mlops:
                    self.mlops_event.log_event_started(
                        "server.wait", event_value=str(self.round_idx)
                    )

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        logging.info("global_model_params = {}".format(global_model_params))
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, data_silo_index
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(data_silo_index))
        self.send_message(message)
