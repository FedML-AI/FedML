
import base64
import json
import logging
import time

import fedml
from ..comm_utils.constants import SchedulerConstants
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from ....core.mlops.mlops_configs import MLOpsConfigs
from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ..scheduler_core.compute_cache_manager import ComputeCacheManager
from ..scheduler_core.ota_upgrade import FedMLOtaUpgrade
from .deploy_job_launcher import FedMLDeployJobLauncher
from ..scheduler_core.general_constants import GeneralConstants
from ..scheduler_core.scheduler_base_protocol_manager import FedMLSchedulerBaseProtocolManager
from abc import ABC, abstractmethod


class FedMLBaseMasterProtocolManager(FedMLSchedulerBaseProtocolManager, ABC):
    def __init__(self, args, agent_config=None):
        FedMLSchedulerBaseProtocolManager.__init__(self, args, agent_config=agent_config, is_master=True)

        self.async_check_timeout = 0
        self.enable_async_cluster = False
        self.request_json = None
        self.run_edge_ids = dict()
        self.version = fedml.get_env_version()
        self.args = args
        self.run_id = None
        self.edge_id = args.edge_id
        self.server_agent_id = args.edge_id
        self.current_device_id = args.current_device_id
        self.unique_device_id = args.unique_device_id
        self.agent_config = agent_config
        self.topic_start_train = None
        self.topic_stop_train = None
        self.topic_complete_job = None
        self.topic_report_status = None
        self.topic_ota_msg = None
        self.topic_response_device_info = None
        self.topic_request_device_info_from_mlops = None
        self.topic_requesst_job_status = None
        self.topic_requesst_device_status_in_job = None
        self.topic_send_training_request_to_edges = None
        self.run_as_cloud_agent = False
        self.run_as_cloud_server = False
        self.run_as_edge_server_and_agent = False
        self.run_as_cloud_server_and_agent = False
        self.enable_simulation_cloud_agent = False
        self.use_local_process_as_cloud_server = False
        self.ota_upgrade = FedMLOtaUpgrade(edge_id=args.edge_id)
        self.running_request_json = dict()
        self.start_request_json = None
        self.deploy_job_launcher = FedMLDeployJobLauncher()

    @abstractmethod
    def generate_topics(self):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # The topic for stopping training
        self.topic_start_train = "mlops/flserver_agent_" + str(self.edge_id) + "/start_train"

        # The topi for stopping training
        self.topic_stop_train = "mlops/flserver_agent_" + str(self.edge_id) + "/stop_train"

        # The topic for completing job
        self.topic_complete_job = GeneralConstants.get_topic_complete_job(self.edge_id)

        # The topic for reporting current device status.
        self.topic_report_status = "mlops/report_device_status"

        # The topic for OTA messages from the MLOps.
        self.topic_ota_msg = "mlops/flserver_agent_" + str(self.edge_id) + "/ota"

        # The topic for requesting device info from the client.
        self.topic_response_device_info = "client/server/response_device_info/" + str(self.edge_id)

        # The topic for requesting device info from mlops.
        self.topic_request_device_info_from_mlops = f"deploy/mlops/master_agent/request_device_info/{self.edge_id}"

        # The topic for getting job status from the status center.
        self.topic_requesst_job_status = f"anywhere/master_agent/request_job_status/{self.edge_id}"

        # The topic for getting device status of job from the status center.
        self.topic_requesst_device_status_in_job = f"anywhere/master_agent/request_device_status_in_job/{self.edge_id}"

        # The topic for reporting online status
        self.topic_active = "flserver_agent/active"

        # The topic for last-will messages.
        self.topic_last_will = "flserver_agent/last_will_msg"

        # Subscribe topics for starting train, stopping train and fetching client status.
        self.subscribed_topics.clear()
        self.add_subscribe_topic(self.topic_start_train)
        self.add_subscribe_topic(self.topic_stop_train)
        self.add_subscribe_topic(self.topic_complete_job)
        self.add_subscribe_topic(self.topic_report_status)
        self.add_subscribe_topic(self.topic_ota_msg)
        self.add_subscribe_topic(self.topic_response_device_info)
        self.add_subscribe_topic(self.topic_request_device_info_from_mlops)
        self.add_subscribe_topic(self.topic_requesst_job_status)
        self.add_subscribe_topic(self.topic_requesst_device_status_in_job)

    @abstractmethod
    def add_protocol_handler(self):
        # Add the message listeners for all topics, the following is an example.
        # self.add_message_listener(self.topic_start_train, self.callback_start_train)
        # Add the message listeners for all topics
        self.add_message_listener(self.topic_start_train, self.callback_start_train)
        self.add_message_listener(self.topic_stop_train, self.callback_stop_train)
        self.add_message_listener(self.topic_complete_job, self.callback_complete_job)
        self.add_message_listener(self.topic_ota_msg, FedMLBaseMasterProtocolManager.callback_server_ota_msg)
        self.add_message_listener(self.topic_report_status, self.callback_report_current_status)
        self.add_message_listener(self.topic_response_device_info, self.callback_response_device_info)
        self.add_message_listener(self.topic_request_device_info_from_mlops,
                                  self.callback_request_device_info_from_mlops)
        self.add_message_listener(self.topic_requesst_job_status, self.callback_request_job_status)
        self.add_message_listener(self.topic_requesst_device_status_in_job, self.callback_request_device_status_in_job)

    @abstractmethod
    def _get_job_runner_manager(self):
        return None

    @abstractmethod
    def _init_extra_items(self):
        pass

    def add_subscribe_topic(self, topic):
        self.subscribed_topics.append(topic)

    def on_agent_communication_connected(self, mqtt_client_object):
        super().on_agent_communication_connected(mqtt_client_object)

        if self.run_as_cloud_server:
            # Start the FedML cloud server
            message_bytes = self.args.runner_cmd.encode("ascii")
            base64_bytes = base64.b64decode(message_bytes)
            payload = base64_bytes.decode("ascii")
            self.receive_message_json(self.topic_start_train, payload)

    def callback_start_train(self, topic=None, payload=None):
        # Fetch config from MLOps
        # noinspection PyBroadException

        try:
            MLOpsConfigs.fetch_all_configs()
        except Exception:
            pass

        # Parse the parameters
        # [NOTES] Example Request JSON:
        # https://fedml-inc.larksuite.com/wiki/ScnIwUif9iupbjkYS0LuBrd6sod#WjbEdhYrvogmlGxKTOGu98C6sSb
        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["runId"]
        run_id_str = str(run_id)

        # Process the log when running in the edge server mode.
        if self.run_as_edge_server_and_agent or self.enable_simulation_cloud_agent:
            # Start log processor for current run
            self.args.run_id = run_id
            self.args.edge_id = self.edge_id
            MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, self.edge_id, SchedulerConstants.get_log_source(request_json))
        # Process the log when running in the cloud agent mode.
        elif self.run_as_cloud_agent:
            # Start log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, request_json.get("server_id", "0"), SchedulerConstants.get_log_source(request_json)
            )
        # Process the log when running in the cloud server mode.
        elif self.run_as_cloud_server:
            # Parse the parameters.
            self.server_agent_id = request_json.get("cloud_agent_id", self.edge_id)
            run_id = request_json["runId"]
            run_id_str = str(run_id)

            # Start log processor for current run.
            self.args.run_id = run_id
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, self.edge_id, SchedulerConstants.get_log_source(request_json))

        # Print the payload
        logging.info("callback_start_train payload: {}".format(payload))
        logging.info(
            f"FedMLDebug - run id {run_id}, Receive at callback_start_train: topic ({topic}), payload ({payload})"
        )

        # Save the parameters
        self.start_request_json = payload
        self.run_id = run_id
        self.request_json = request_json
        self.running_request_json[run_id_str] = request_json
        edge_id_list = request_json.get("edgeids", list())
        self.run_edge_ids[run_id_str] = edge_id_list

        # report server running status to master agent
        if not self.run_as_cloud_server and not self.run_as_cloud_agent:
            self.mlops_metrics.report_server_id_status(
                run_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_STARTING, edge_id=self.edge_id,
                server_id=self.edge_id, server_agent_id=self.edge_id, running_json=payload)

        # Start server with multiprocessing mode
        if self.run_as_edge_server_and_agent or self.enable_simulation_cloud_agent:
            self.init_job_task(request_json)

            self.args.run_id = run_id

            self._get_job_runner_manager().start_job_runner(
                run_id, request_json, args=self.args, edge_id=self.edge_id,
                sender_message_queue=self.message_center.get_sender_message_queue(),
                listener_message_queue=self.get_listener_message_queue(),
                status_center_queue=self.get_status_queue(),
                communication_manager=self.get_listener_communication_manager(),
                process_name=GeneralConstants.get_launch_master_job_process_name(run_id, self.edge_id)
            )

            process = self._get_job_runner_manager().get_runner_process(run_id)
            if process is not None:
                GeneralConstants.save_run_process(run_id, process.pid, is_master=True)

            self.send_status_msg_to_edges(edge_id_list, run_id, self.edge_id)
        elif self.run_as_cloud_agent:
            self.init_job_task(request_json)

            server_id = request_json.get("server_id", self.edge_id)
            self._get_job_runner_manager().start_job_runner(
                run_id, request_json, args=self.args, edge_id=self.edge_id,
                sender_message_queue=self.message_center.get_sender_message_queue(),
                listener_message_queue=self.get_listener_message_queue(),
                status_center_queue=self.get_status_queue(),
                communication_manager=self.get_listener_communication_manager(),
                master_agent_instance=self.generate_agent_instance(),
                should_start_cloud_server=True,
                use_local_process_as_cloud_server=self.use_local_process_as_cloud_server,
                process_name=GeneralConstants.get_launch_master_job_process_name(run_id, server_id)
            )

            process = self._get_job_runner_manager().get_runner_process(run_id, is_cloud_server=True)
            if process is not None:
                GeneralConstants.save_run_process(run_id, process.pid, is_master=True)
        elif self.run_as_cloud_server:
            self.server_agent_id = request_json.get("cloud_agent_id", self.edge_id)
            self.start_request_json = json.dumps(request_json)
            server_id = request_json.get("server_id", self.edge_id)
            run_id = request_json["runId"]
            run_id_str = str(run_id)

            self.init_job_task(request_json)

            self.args.run_id = run_id

            self._get_job_runner_manager().start_job_runner(
                run_id, request_json, args=self.args, edge_id=self.edge_id,
                sender_message_queue=self.message_center.get_sender_message_queue(),
                listener_message_queue=self.get_listener_message_queue(),
                status_center_queue=self.get_status_queue(),
                communication_manager=self.get_listener_communication_manager(),
                process_name=GeneralConstants.get_launch_master_job_process_name(run_id, server_id)
            )

            self.send_status_msg_to_edges(edge_id_list, run_id, server_id)

    def callback_stop_train(self, topic, payload, use_payload=None):
        # Print the payload
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        # Parse the parameters.
        request_json = json.loads(payload)
        run_id = request_json.get("runId", None)
        run_id = request_json.get("id", None) if run_id is None else run_id
        run_id_str = str(run_id)
        edge_ids = request_json.get("edgeids", None)
        server_id = request_json.get("serverId", None)
        if server_id is None:
            server_id = request_json.get("server_id", None)
        server_agent_id = server_id

        # Cleanup the cached object
        if self.running_request_json.get(run_id_str, None) is not None:
            self.running_request_json.pop(run_id_str)

        # If it is the cloud agent, then forward the stopping request to the corresponding cloud server.
        if self.run_as_cloud_agent:
            server_agent_id = self.edge_id
            topic_stop_train_to_cloud_server = f"mlops/flserver_agent_{server_id}/stop_train"
            self.message_center.send_message(topic_stop_train_to_cloud_server, payload)

            time.sleep(2)
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, server_id)
            self._get_job_runner_manager().stop_job_runner(
                run_id, args=self.args, server_id=server_id, request_json=None,
                run_as_cloud_agent=self.run_as_cloud_agent, run_as_cloud_server=self.run_as_cloud_server,
                use_local_process_as_cloud_server=self.use_local_process_as_cloud_server)
            self.generate_status_report(run_id, server_id, server_agent_id=server_agent_id). \
                report_server_id_status(run_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_KILLED,
                                        edge_id=server_id, server_id=server_id)
            return

        # Reset all edge status and server status
        for iter_edge_id in edge_ids:
            self.generate_status_report(run_id, iter_edge_id, server_agent_id=server_agent_id).\
                report_client_id_status(iter_edge_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_KILLED,
                                        run_id=run_id, server_id=server_id)

        # To be compatible to the previous version of edge devices, we just send the stopping train message to edges.
        # Currently, the latest version of edge devices don't need to process the stopping train message.
        self.send_training_stop_request_to_edges(edge_ids, payload=payload, run_id=run_id)

    def callback_complete_job(self, topic, payload):
        # Parse the parameters.
        request_json = json.loads(payload)
        run_id = request_json.get("runId", None)
        run_id = request_json.get("id", None) if run_id is None else run_id
        run_id_str = str(run_id)
        server_id = request_json.get("serverId", None)
        if server_id is None:
            server_id = request_json.get("server_id", None)

        self._process_job_complete_status(run_id, server_id, request_json)

    def _process_job_complete_status(self, run_id, server_id, complete_payload):
        # Complete the job runner
        self._get_job_runner_manager().complete_job_runner(
            run_id, args=self.args, server_id=server_id, request_json=complete_payload,
            run_as_cloud_agent=self.run_as_cloud_agent, run_as_cloud_server=self.run_as_cloud_server,
            use_local_process_as_cloud_server=self.use_local_process_as_cloud_server)

    def callback_run_logs(self, topic, payload):
        run_id = str(topic).split('/')[-1]
        run_id_str = str(run_id)
        self._get_job_runner_manager().callback_run_logs(run_id, topic, payload)

    def callback_run_metrics(self, topic, payload):
        run_id = str(topic).split('/')[-1]
        run_id_str = str(run_id)
        self._get_job_runner_manager().callback_run_metrics(run_id, topic, payload)

    def callback_edge_status(self, topic, payload):
        self.send_status_message(topic, payload)

    def callback_report_current_status(self, topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        if self.run_as_edge_server_and_agent:
            self.send_agent_active_msg(self.edge_id)
        elif self.run_as_cloud_agent:
            self.send_agent_active_msg(self.edge_id)
        elif self.run_as_cloud_server:
            pass

    @staticmethod
    def callback_server_ota_msg(topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        cmd = request_json["cmd"]

        if cmd == GeneralConstants.FEDML_OTA_CMD_UPGRADE:
            # noinspection PyBroadException
            try:
                FedMLOtaUpgrade.process_ota_upgrade_msg()
                # Process(target=FedMLServerRunner.process_ota_upgrade_msg).start()
                raise Exception("After upgraded, restart runner...")
            except Exception as e:
                pass
        elif cmd == GeneralConstants.FEDML_OTA_CMD_RESTART:
            raise Exception("Restart runner...")

    def callback_response_device_info(self, topic, payload):
        # Parse payload
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", 0)
        context = payload_json.get("context", None)
        master_device_id = payload_json.get("master_device_id", 0)
        slave_device_id = payload_json.get("slave_device_id", 0)
        slave_device_id_list = payload_json.get("slave_device_id_list", 0)
        edge_id = payload_json.get("edge_id", 0)
        device_info = payload_json.get("edge_info", 0)
        device_info["master_device_id"] = master_device_id
        device_info["slave_device_id"] = slave_device_id
        device_info["slave_device_id_list"] = slave_device_id_list
        run_id_str = str(run_id)

        # Put device info into a multiprocessing queue so master runner checks if all edges are ready
        if context is None:
            self._get_job_runner_manager().put_run_edge_device_info_to_queue(run_id, edge_id, device_info)

            # if self.run_edge_device_info_global_queue is None:
            #     self.run_edge_device_info_global_queue = Array('i', list())
            #
            # self.run_edge_device_info_global_queue[len(self.run_edge_device_info_global_queue)] =  \
            #     {"timestamp": time.time(), "edge_id": edge_id, "device_info": device_info}

            request_json = self.running_request_json.get(str(run_id), None)
            if request_json is not None:
                self.deploy_job_launcher.check_model_device_ready_and_deploy(
                    request_json, run_id, master_device_id, slave_device_id, run_edge_ids=self.run_edge_ids)

    def callback_request_device_info_from_mlops(self, topic, payload):
        self.response_device_info_to_mlops(topic, payload)

    def callback_request_job_status(self, topic, payload):
        self.response_job_status(topic, payload)

    def callback_request_device_status_in_job(self, topic, payload):
        self.response_device_status_in_job(topic, payload)

    def callback_proxy_unknown_messages(self, run_id, topic, payload):
        self._get_job_runner_manager().callback_proxy_unknown_messages(run_id, topic, payload)

    def process_extra_queues(self, extra_queues):
        self.rebuild_status_center(extra_queues[0])

    def generate_protocol_manager(self):
        message_status_runner = self._generate_protocol_manager_instance(
            self.args, agent_config=self.agent_config
        )
        message_status_runner.async_check_timeout = self.async_check_timeout
        message_status_runner.enable_async_cluster = self.enable_async_cluster
        message_status_runner.request_json = self.request_json
        message_status_runner.run_edge_ids = self.run_edge_ids
        message_status_runner.version = self.version
        message_status_runner.message_center_name = self.message_center_name
        message_status_runner.run_id = self.run_id
        message_status_runner.edge_id = self.edge_id
        message_status_runner.server_agent_id = self.server_agent_id
        message_status_runner.current_device_id = self.current_device_id
        message_status_runner.unique_device_id = self.unique_device_id
        message_status_runner.subscribed_topics = self.subscribed_topics
        message_status_runner.run_as_cloud_agent = self.run_as_cloud_agent
        message_status_runner.run_as_cloud_server = self.run_as_cloud_server
        message_status_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
        message_status_runner.run_as_cloud_server_and_agent = self.run_as_cloud_server_and_agent
        message_status_runner.enable_simulation_cloud_agent = self.enable_simulation_cloud_agent
        message_status_runner.use_local_process_as_cloud_server = self.use_local_process_as_cloud_server
        message_status_runner.running_request_json = self.running_request_json
        message_status_runner.start_request_json = self.start_request_json
        message_status_runner.user_name = self.user_name
        message_status_runner.status_queue = self.get_status_queue()

        return message_status_runner

    def response_job_status(self, topic, payload):
        payload_json = json.loads(payload)
        if self.mlops_metrics is not None:
            run_id = payload_json.get("run_id", None)
            edge_id = payload_json.get("edge_id", None)
            if run_id is None or edge_id is None:
                return
            response_topic = f"master_agent/somewhere/response_job_status/{edge_id}"
            response_payload = {
                "run_id": run_id,
                "master_agent": self.edge_id,
                "edge_id": edge_id,
                "job_status": ComputeCacheManager.get_instance().get_status_cache().get_job_status(),
                "fedml_version": fedml.__version__
            }
            self.mlops_metrics.report_json_message(response_topic, json.dumps(response_payload))

    def response_device_status_in_job(self, topic, payload):
        payload_json = json.loads(payload)
        if self.mlops_metrics is not None:
            run_id = payload_json.get("run_id", None)
            edge_id = payload_json.get("edge_id", None)
            if run_id is None or edge_id is None:
                return
            response_topic = f"master_agent/somewhere/response_device_status_in_job/{edge_id}"
            response_payload = {
                "run_id": run_id,
                "master_agent": self.edge_id,
                "edge_id": edge_id,
                "device_status_in_job":
                    ComputeCacheManager.get_instance().get_status_cache().get_device_status_in_job(run_id, edge_id),
                "fedml_version": fedml.__version__
            }
            self.mlops_metrics.report_json_message(response_topic, json.dumps(response_payload))

    def response_device_info_to_mlops(self, topic, payload):
        response_topic = f"deploy/master_agent/mlops/response_device_info"
        if self.mlops_metrics is not None:
            response_payload = {"run_id": self.run_id, "master_agent_device_id": self.edge_id,
                                "fedml_version": fedml.__version__, "edge_id": self.edge_id}
            self.mlops_metrics.report_json_message(response_topic, json.dumps(response_payload))

    def init_job_task(self, request_json):
        run_id = request_json["runId"]
        run_config = request_json["run_config"]
        edge_ids = request_json["edgeids"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", None)
        server_id = request_json["server_id"]
        if self.run_as_cloud_agent:
            server_id = self.edge_id

        self.setup_listeners_for_edge_status(run_id, edge_ids, server_id)
        self.setup_listener_for_run_metrics(run_id)
        self.setup_listener_for_run_logs(run_id)

    def setup_listeners_for_edge_status(self, run_id, edge_ids, server_id):
        if self.run_as_cloud_agent:
            return
        edge_status_topic = "fl_client/flclient_agent_" + str(server_id) + "/status"
        payload = {"run_id": run_id, "init_all_edge_id_list": edge_ids, "init_server_id": server_id}
        self.callback_edge_status(edge_status_topic, json.dumps(payload))

        for edge_id in edge_ids:
            edge_status_topic = "fl_client/flclient_agent_" + str(edge_id) + "/status"
            self.add_message_listener(edge_status_topic, self.callback_edge_status)
            self.subscribe_msg(edge_status_topic)

    def remove_listeners_for_edge_status(self, edge_ids=None):
        if self.run_as_cloud_agent:
            return

        if edge_ids is None:
            edge_ids = self.request_json["edgeids"]

        for edge_id in edge_ids:
            edge_status_topic = "fl_client/flclient_agent_" + str(edge_id) + "/status"
            self.unsubscribe_msg(edge_status_topic)

    def setup_listener_for_run_metrics(self, run_id):
        metric_topic = f"fedml_slave/fedml_master/metrics/{run_id}"
        self.add_message_listener(metric_topic, self.callback_run_metrics)
        self.subscribe_msg(metric_topic)

    def remove_listener_for_run_metrics(self, run_id):
        metric_topic = f"fedml_slave/fedml_master/metrics/{run_id}"
        self.unsubscribe_msg(metric_topic)

    def setup_listener_for_run_logs(self, run_id):
        logs_topic = f"fedml_slave/fedml_master/logs/{run_id}"
        self.add_message_listener(logs_topic, self.callback_run_logs)
        self.subscribe_msg(logs_topic)

    def remove_listener_for_run_logs(self, run_id):
        logs_topic = f"fedml_slave/fedml_master/logs/{run_id}"
        self.unsubscribe_msg(logs_topic)

    def send_training_stop_request_to_edges(
            self, edge_id_list, payload=None, run_id=0):
        if payload is None:
            payload_obj = {"runId": run_id, "edgeids": edge_id_list}
            payload = json.dumps(payload_obj)

        for edge_id in edge_id_list:
            topic_stop_train = "flserver_agent/" + str(edge_id) + "/stop_train"
            logging.info("stop_train: send topic " + topic_stop_train)
            self.message_center.send_message(topic_stop_train, payload)

    def send_training_stop_request_to_specific_edge(self, edge_id, payload):
        topic_stop_train = "flserver_agent/" + str(edge_id) + "/stop_train"
        logging.info("stop_train: send topic " + topic_stop_train)
        self.message_center.send_message(topic_stop_train, payload)

    def send_training_stop_request_to_cloud_server(self, edge_id, payload):
        topic_stop_train = "mlops/flserver_agent_" + str(edge_id) + "/stop_train"
        logging.info("stop_train: send topic " + topic_stop_train)
        self.message_center.send_message(topic_stop_train, payload)

    def send_status_check_msg(self, run_id, edge_id, server_id, context=None):
        topic_status_check = f"server/client/request_device_info/{edge_id}"
        payload = {"server_id": server_id, "run_id": run_id}
        if context is not None:
            payload["context"] = context
        self.message_center.send_message(topic_status_check, json.dumps(payload))

    def send_status_msg_to_edges(self, edge_id_list, run_id, server_id, context=None):
        # Send status message to all edges
        for edge_id in edge_id_list:
            self.send_status_check_msg(run_id, edge_id, server_id, context=context)

    def report_exception_status(self, run_id):
        self.mlops_metrics.report_job_status(run_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_EXCEPTION)

    @staticmethod
    def get_start_train_topic_with_edge_id(edge_id):
        return "mlops/flserver_agent_" + str(edge_id) + "/start_train"

    @abstractmethod
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return None

    def start_master_server_instance(self, payload):
        super().on_agent_communication_connected(None)

        self.receive_message_json(self.topic_start_train, payload)

