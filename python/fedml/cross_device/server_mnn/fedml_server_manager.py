import json
import time

from .message_define import MyMessage
from .utils import write_tensor_dict_to_mnn
from ...core.distributed.communication.message import Message
from ...core.distributed.server.server_manager import ServerManager
from ...mlops import MLOpsMetrics, MLOpsProfilerEvent
from ...utils.logging import logger


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
        logger.info(
            "self.global_model_file_path = {}".format(self.global_model_file_path)
        )
        logger.info(
            "self.model_file_cache_folder = {}".format(self.model_file_cache_folder)
        )

        self.client_online_mapping = {}
        self.client_real_ids = json.loads(args.client_id_list)

        self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self.com_manager_status)
        self.aggregator.set_mlops_logger(self.mlops_metrics)
        self.start_running_time = 0.0
        self.aggregated_model_url = None
        self.event_sdk = MLOpsProfilerEvent(self.args)

    def run(self):
        # notify MLOps with RUNNING status
        self.mlops_metrics.report_server_training_status(
            self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING
        )

        super().run()

    def start_train(self):
        start_train_json = {
            "edges": [
                {
                    "device_id": "647e593ab312c934",
                    "os_type": "Android",
                    "id": self.args.client_id_list,
                }
            ],
            "starttime": 1650701355944,
            "url": "http://fedml-server-agent-svc.fedml-aggregator-dev.svc.cluster.local:5001/api/start_run",
            "edgeids": self.args.client_id_list,
            "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6NzcsImFjY291bnQiOiJ1bHRyYXoyIiwibG9naW5UaW1lIjoiMTY1MDY5MDUwNzc2MiIsImV4cCI6MH0.DZBzihroMcJA8BVXNDJo8fktz31MsEsLUaMdUnS726g",
            "urls": [],
            "userids": ["77"],
            "name": "dinner_sheep",
            "runId": 189,
            "id": 169,
            "projectid": "169",
            "run_config": {
                "configName": "fedml_fedavg",
                "userId": 77,
                "model_config": {"modelName": "lenet_mnist"},
                "packages_config": {
                    "server": "package.zip",
                    "linuxClient": "",
                    "serverUrl": "https://fedml.s3.us-west-1.amazonaws.com/1650701256822package.zip",
                    "linuxClientUrl": "",
                    "androidClient": "",
                    "androidClientUrl": "",
                    "androidClientVersion": "0",
                },
                "data_config": {
                    "privateLocalData": "",
                    "syntheticData": "",
                    "syntheticDataUrl": "",
                },
                "hyperparameters_config": {
                    "client_learning_rate": 0.001,
                    "partition_method": "homo",
                    "train_batch_size": 8,
                    "client_optimizer": "sgd",
                    "comm_round": 3,
                    "local_epoch": 1,
                    "dataset": "mnist",
                    "communication_backend": "MQTT_S3",
                    "data_silo_num_in_total": 1,
                    "client_num_in_total": 1,
                    "client_num_per_round": 1,
                },
            },
            "timestamp": "1650701355951",
        }
        for client_id in self.client_real_ids:
            logger.info("com_manager_status - client_id = {}".format(client_id))
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
        logger.info(
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

        self.event_sdk.log_event_started("server.wait")

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

        self.event_sdk.log_event_started("aggregator.wait-online")

        all_client_is_online = True
        for client_id in self.client_real_ids:
            if not self.client_online_mapping.get(str(client_id), False):
                all_client_is_online = False
                break

        logger.info(
            "sender_id = %d, all_client_is_online = %s"
            % (msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            # send initialization message to all clients to start training
            self.send_init_msg()

        self.event_sdk.log_event_ended("aggregator.wait-online")

    def handle_message_receive_model_from_client(self, msg_params):
        self.event_sdk.log_event_ended("comm_c2s", event_edge_id=0)

        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        logger.info("model_params = {}".format(model_params))

        self.event_sdk.log_event_started("aggregator.global-aggregate")

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logger.info("b_all_received = %s " % str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            write_tensor_dict_to_mnn(self.global_model_file_path, global_model_params)

            self.event_sdk.log_event_started("aggregate")
            self.aggregator.test_on_server_for_all_clients(self.global_model_file_path)
            self.event_sdk.log_event_ended("aggregate")

            # send round info to the MQTT backend
            round_info = {
                "run_id": self.args.run_id,
                "round_index": self.round_idx,
                "total_rounds": self.round_num,
                "running_time": round(time.time() - self.start_running_time, 4),
            }
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

            model_info = {
                "run_id": self.args.run_id,
                "round_idx": self.round_idx + 1,
                "global_aggregated_model_s3_address": self.aggregated_model_url,
            }
            self.mlops_metrics.report_aggregated_model_info(model_info)
            self.aggregated_model_url = None

            self.round_idx += 1
            if self.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                self.mlops_metrics.report_server_id_status(
                    self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_FINISHED
                )
                self.finish()
                return

        self.event_sdk.log_event_ended("wait")

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        logger.info("global_model_params = {}".format(global_model_params))
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, data_silo_index
    ):
        logger.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(data_silo_index))
        self.send_message(message)
