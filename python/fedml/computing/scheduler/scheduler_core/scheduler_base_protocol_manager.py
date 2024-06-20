
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
import uuid
import fedml
from ..comm_utils.run_process_utils import RunProcessUtils
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from ....core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ....core.mlops.mlops_metrics import MLOpsMetrics
from ..comm_utils import sys_utils
from ..scheduler_core.message_center import FedMLMessageCenter
from ..scheduler_core.status_center import FedMLStatusCenter
from .account_manager import FedMLAccountManager
from .general_constants import GeneralConstants
from abc import ABC, abstractmethod


class FedMLSchedulerBaseProtocolManager(FedMLMessageCenter, FedMLStatusCenter, ABC):

    def __init__(self, args, agent_config=None, is_master=False):
        FedMLMessageCenter.__init__(self)
        FedMLStatusCenter.__init__(self)
        self.request_json = None
        self.version = fedml.get_env_version()
        self.args = args
        self.is_master_agent = is_master
        self.message_status_runner = None
        self.message_center = None
        self.status_center = None
        self.message_center_name = "master_agent" if is_master else "slave_agent"
        self.run_id = None
        self.edge_id = args.edge_id
        self.general_edge_id = None
        self.server_agent_id = args.edge_id
        self.current_device_id = args.current_device_id
        self.unique_device_id = args.unique_device_id
        self.agent_config = agent_config
        self.topic_active = None
        self.topic_last_will = None
        self.communication_mgr = None
        self.subscribed_topics = list()
        self.mlops_metrics = None
        self.status_reporter = None
        self.user_name = args.user_name
        self.parent_agent = None

        fedml._init_multiprocessing()

    def generate_topics(self):
        # generate the subscribed topics.
        self.subscribed_topics.clear()
        # self.subscribed_topics.append(self.topic_start_train)

    def add_protocol_handler(self):
        # Add the message listeners for all topics, the following is an example.
        # self.add_message_listener(self.topic_start_train, self.callback_start_train)
        pass

    def initialize(
            self, communication_manager=None, sender_message_queue=None,
            status_center_queue=None, sender_message_event=None
    ):
        # Generate the message topics
        self.generate_topics()

        # Setup MQTT connection
        if communication_manager is None:
            self.communication_mgr = MqttManager(
                self.agent_config["mqtt_config"]["BROKER_HOST"],
                self.agent_config["mqtt_config"]["BROKER_PORT"],
                self.agent_config["mqtt_config"]["MQTT_USER"],
                self.agent_config["mqtt_config"]["MQTT_PWD"],
                self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
                f"FedML_Agent_Daemon_@{self.user_name}@_@{self.current_device_id}@_@{str(uuid.uuid4())}@",
                self.topic_last_will,
                json.dumps({"ID": self.edge_id, "status": GeneralConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE})
            )
        else:
            self.communication_mgr = communication_manager

        # Add the message listeners for all topics
        self.add_protocol_handler()

        # Start the message center to process edge related messages.
        if sender_message_queue is None:
            self.setup_message_center()
            sender_message_event = self.message_center.get_sender_message_event()
        else:
            self.rebuild_message_center(sender_message_queue)

        # Setup the message listener queue
        self.setup_listener_message_queue()

        # Start the status center to process edge related status.
        if status_center_queue is None:
            self.start_status_listener_center(sender_message_event=sender_message_event)
        else:
            self.set_status_queue(status_center_queue)
            self.rebuild_status_center(status_center_queue)

        # Start the message center for listener
        self.start_listener(sender_message_queue=self.message_center.get_sender_message_queue(),
                            sender_message_event=sender_message_event,
                            agent_config=self.agent_config,
                            message_center_name=self.message_center_name,
                            extra_queues=[self.get_status_queue()])

        # Init extra items, e.g. database, recovery, etc.
        self._init_extra_items()

        # Setup MQTT connected listener
        self.communication_mgr.add_connected_listener(self.on_agent_communication_connected)
        self.communication_mgr.add_disconnected_listener(self.on_agent_communication_disconnected)

    def start(self):
        # Start MQTT message loop
        try:
            self.communication_mgr.connect()
            self.communication_mgr.loop_forever()
        except Exception as e:
            if str(e) == "Restarting after upgraded...":
                logging.info("Restarting after upgraded...")
            else:
                logging.info("Server tracing: {}".format(traceback.format_exc()))

        finally:
            logging.info(f"Protocol manager is about to exit, pid: {os.getpid()}")

            FedMLAccountManager.write_login_failed_file(is_client=not self.is_master_agent)

            self.stop()

            time.sleep(5)
            sys_utils.cleanup_all_fedml_server_login_processes(
                GeneralConstants.MASTER_LOGIN_PROGRAM if self.is_master_agent else GeneralConstants.SLAVE_LOGIN_PROGRAM,
                clean_process_group=False)
            sys.exit(1)

    def stop(self, kill_process=False):
        if self.communication_mgr is not None:
            # noinspection PyBroadException
            try:
                for topic in self.subscribed_topics:
                    self.communication_mgr.unsubscribe_msg(topic)
            except Exception:
                pass

            self.communication_mgr.loop_stop()
            self.communication_mgr.disconnect()

        if kill_process:
            self.post_status_center_stopping_message()
            self.release_message_center()
            RunProcessUtils.kill_process(os.getppid(), exclude_current_pid=True)

    @abstractmethod
    def _init_extra_items(self):
        pass

    def on_agent_communication_connected(self, mqtt_client_object):
        # Setup MQTT message passthrough listener for all messages
        self.communication_mgr.add_message_passthrough_listener(self.listener_message_passthrough_dispatch_center)

        # Subscribe topics for starting train, stopping train and fetching client status.
        for topic in self.subscribed_topics:
            self.communication_mgr.subscribe_msg(topic)

        # Broadcast the first active message.
        self.send_agent_active_msg(self.edge_id)
        if self.general_edge_id is not None:
            self.send_agent_active_msg(self.general_edge_id)

        # Echo results
        MLOpsRuntimeLog.get_instance(self.args).enable_show_log_to_stdout()
        self.print_connected_info()
        MLOpsRuntimeLog.get_instance(self.args).enable_show_log_to_stdout(enable=True)

    @abstractmethod
    def print_connected_info(self):
        print("\nCongratulations, your device is connected to the FedML MLOps platform successfully!")
        print(
            "Your FedML Edge ID is " + str(self.edge_id) + ", unique device ID is "
            + str(self.unique_device_id)
        )

    def on_agent_communication_disconnected(self, mqtt_client_object):
        pass

    def setup_message_center(self):
        if self.message_center is not None:
            return

        self.message_center = FedMLMessageCenter(agent_config=self.agent_config)
        self.message_center.start_sender(message_center_name=self.message_center_name)

        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self)
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = self.edge_id
        self.mlops_metrics.server_agent_id = self.server_agent_id

    def send_message_json(self, topic, payload):
        self.message_center.send_message_json(topic, payload)

    def rebuild_message_center(self, message_center_queue):
        self.message_center = FedMLMessageCenter(sender_message_queue=message_center_queue)

        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self)
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = self.edge_id
        self.mlops_metrics.server_agent_id = self.server_agent_id

    def release_message_center(self):
        try:
            self.stop_message_center()

            if self.message_center is not None:
                self.message_center.stop_message_center()
                self.message_center = None

        except Exception as e:
            logging.error(
                f"Failed to release the message center with Exception {e}. "
                f"Traceback: {traceback.format_exc()}")
            pass

    def release_status_center(self):
        try:
            self.stop_status_center()

            if self.status_center is not None:
                self.status_center.stop_status_center()
                self.status_center = None

        except Exception as e:
            logging.error(
                f"Failed to release the status center with Exception {e}. "
                f"Traceback: {traceback.format_exc()}")
            pass

    def start_status_listener_center(self, sender_message_event=None):
        self.start_status_center(
            sender_message_center_queue=self.message_center.get_sender_message_queue(),
            listener_message_center_queue=self.get_listener_message_queue(),
            sender_message_event=sender_message_event,
            is_slave_agent=not self.is_master_agent
        )

        if self.status_reporter is None:
            self.status_reporter = MLOpsMetrics()
        self.status_reporter.set_messenger(self, send_message_func=self.send_status_message)
        self.status_reporter.run_id = self.run_id
        self.status_reporter.edge_id = self.edge_id
        self.status_reporter.server_agent_id = self.server_agent_id

    def rebuild_status_center(self, status_center_queue):
        self.status_center = FedMLStatusCenter(message_queue=status_center_queue)
        self.status_center.is_deployment_status_center = self.is_deployment_status_center

        if self.status_reporter is None:
            self.status_reporter = MLOpsMetrics()
        self.status_reporter.set_messenger(self.status_center, send_message_func=self.status_center.send_status_message)
        self.status_reporter.run_id = self.run_id
        self.status_reporter.edge_id = self.edge_id
        self.status_reporter.server_agent_id = self.server_agent_id

    def process_extra_queues(self, extra_queues):
        pass

    def generate_status_report(self, run_id, edge_id, server_agent_id=None):
        status_reporter = MLOpsMetrics()
        status_reporter.set_messenger(self, send_message_func=self.send_status_message)
        status_reporter.run_id = run_id
        status_reporter.edge_id = edge_id
        if server_agent_id is not None:
            status_reporter.server_agent_id = server_agent_id
        return status_reporter

    @abstractmethod
    def generate_protocol_manager(self):
        # Generate the protocol manager instance and set the attribute values.
        return None

    def get_message_runner(self):
        if self.message_status_runner is not None:
            return self.message_status_runner

        self.message_status_runner = self.generate_protocol_manager()
        self.message_status_runner.status_queue = self.get_status_queue()

        return self.message_status_runner

    def get_status_runner(self):
        if self.message_status_runner is None:
            self.get_message_runner()
            if self.message_status_runner is not None:
                self.message_status_runner.sender_message_queue = self.message_center.get_sender_message_queue()

        if self.message_status_runner is not None:
            self.message_status_runner.sender_message_queue = self.message_center.get_sender_message_queue()
            return self.message_status_runner

        return None

    def get_protocol_communication_manager(self):
        return self.communication_mgr

    def get_protocol_sender_message_queue(self):
        return self.message_center.get_sender_message_queue()

    def get_protocol_sender_message_event(self):
        return self.message_center.get_sender_message_event()

    def get_protocol_status_center_queue(self):
        return self.get_status_queue()

    def get_subscribed_topics(self):
        return self.subscribed_topics

    def send_agent_active_msg(self, edge_id):
        active_msg = {"ID": edge_id, "status": GeneralConstants.MSG_MLOPS_SERVER_STATUS_IDLE}
        self.message_center.send_message_json(self.topic_active, json.dumps(active_msg))

    def post_status_center_stopping_message(self, run_id=None):
        topic_status_center_stopping = GeneralConstants.FEDML_TOPIC_STATUS_CENTER_STOP
        payload = {"run_id": run_id}
        self.status_reporter.send_message(topic_status_center_stopping, json.dumps(payload))

    def set_parent_agent(self, parent_agent):
        self.parent_agent = parent_agent
