import json
import logging
import os
import platform
import threading
import time
import traceback
import uuid
import multiprocessing
import queue
from os.path import expanduser

import setproctitle

import fedml
from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager
from .general_constants import GeneralConstants
from ..slave.client_constants import ClientConstants
from ....core.mlops.mlops_metrics import MLOpsMetrics
from operator import methodcaller
from .message_common import FedMLMessageEntity, FedMLMessageRecord


class FedMLMessageCenter(object):
    FUNC_SETUP_MESSAGE_CENTER = "setup_message_center"
    FUNC_REBUILD_MESSAGE_CENTER = "rebuild_message_center"
    FUNC_PROCESS_EXTRA_QUEUES = "process_extra_queues"
    ENABLE_SAVE_MESSAGE_TO_FILE = True
    PUBLISH_MESSAGE_RETRY_TIMEOUT = 60 * 1000.0
    PUBLISH_MESSAGE_RETRY_COUNT = 3
    MESSAGE_SENT_RECORDS_FILE = "message-sent-records.log"
    MESSAGE_SENT_SUCCESS_RECORDS_FILE = "message-sent-success-records.log"
    MESSAGE_RECEIVED_RECORDS_FILE = "message-received-records.log"

    def __init__(self, agent_config=None, sender_message_queue=None,
                 listener_message_queue=None, sender_message_event=None):
        self.sender_agent_config = agent_config
        self.listener_agent_config = agent_config
        self.sender_message_queue = sender_message_queue
        self.message_event = sender_message_event
        self.message_center_process = None
        self.sender_mqtt_mgr = None
        self.sender_mlops_metrics = None
        self.sender_mqtt_lock = None
        self.sender_mqtt_is_connected = False
        self.listener_mqtt_mgr = None
        self.listener_topics = list()
        self.listener_payloads = dict()
        self.listener_handler_funcs = dict()
        self.listener_handler_object = None
        self.listener_message_queue = listener_message_queue
        self.listener_message_event = None
        self.listener_message_center_process = None
        self.sender_message_list = list()
        self.receiver_message_list = list()
        self.published_message_ids = list()
        self.retry_sending_count_map = dict()
        self.constants = FedMLMessageCenterConstants()
        self.message_center_name = None

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def on_sender_mqtt_disconnected(self, mqtt_client_object):
        if self.sender_mqtt_lock is None:
            self.sender_mqtt_lock = threading.Lock()

        self.sender_mqtt_lock.acquire()
        self.sender_mqtt_is_connected = False
        self.sender_mqtt_lock.release()

    def on_sender_mqtt_connected(self, mqtt_client_object):
        if self.sender_mlops_metrics is None:
            self.sender_mlops_metrics = MLOpsMetrics()

        self.sender_mlops_metrics.set_messenger(self)

        if self.sender_mqtt_lock is None:
            self.sender_mqtt_lock = threading.Lock()

        self.sender_mqtt_lock.acquire()
        self.sender_mqtt_is_connected = True
        self.sender_mqtt_lock.release()

    def on_sender_mqtt_published(self, mqtt_client_object, obj, mid):
        self.published_message_ids.append({"message_id": mid, "timestamp": time.time_ns()/100.0/1000.0})
        self.save_published_message_record(mid)

    def setup_sender_mqtt_mgr(self):
        if self.sender_mqtt_mgr is not None:
            return

        if self.sender_mqtt_lock is None:
            self.sender_mqtt_lock = threading.Lock()

        self.sender_mqtt_mgr = MqttManager(
            self.sender_agent_config["mqtt_config"]["BROKER_HOST"],
            self.sender_agent_config["mqtt_config"]["BROKER_PORT"],
            self.sender_agent_config["mqtt_config"]["MQTT_USER"],
            self.sender_agent_config["mqtt_config"]["MQTT_PWD"],
            self.sender_agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_MessageCenter_@{}@{}".format(str(os.getpid()), str(uuid.uuid4()))
        )

        self.sender_mqtt_mgr.add_connected_listener(self.on_sender_mqtt_connected)
        self.sender_mqtt_mgr.add_disconnected_listener(self.on_sender_mqtt_disconnected)
        self.sender_mqtt_mgr.add_published_listener(self.on_sender_mqtt_published)
        self.sender_mqtt_mgr.connect()
        self.sender_mqtt_mgr.loop_start()

        if self.sender_mlops_metrics is None:
            self.sender_mlops_metrics = MLOpsMetrics()
        self.sender_mlops_metrics.set_messenger(self)

    def release_sender_mqtt_mgr(self):
        # noinspection PyBroadException
        try:
            if self.sender_mqtt_mgr is not None:
                self.sender_mqtt_mgr.loop_stop()
                self.sender_mqtt_mgr.disconnect()

            self.sender_mqtt_lock.acquire()
            if self.sender_mqtt_mgr is not None:
                self.sender_mqtt_is_connected = False
                self.sender_mqtt_mgr = None
            self.sender_mqtt_lock.release()
        except Exception as e:
            logging.error(
                f"Failed to release sender mqtt manager with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

    def get_sender_message_queue(self):
        return self.sender_message_queue

    def get_sender_message_event(self):
        return self.message_event

    def start_sender(self, message_center_name=None):
        self.sender_message_queue = multiprocessing.Manager().Queue()
        self.message_event = multiprocessing.Event()
        self.message_event.clear()
        process_name = GeneralConstants.get_message_center_sender_process_name(message_center_name)
        message_center = FedMLMessageCenter(agent_config=self.sender_agent_config,
                                            sender_message_queue=self.sender_message_queue)
        if platform.system() == "Windows":
            self.message_center_process = multiprocessing.Process(
                target=message_center.run_sender, args=(
                    self.message_event, self.sender_message_queue,
                    message_center_name, process_name
                )
            )
        else:
            self.message_center_process = fedml.get_process(
                target=message_center.run_sender, args=(
                    self.message_event, self.sender_message_queue,
                    message_center_name, process_name
                )
            )
        self.message_center_process.start()

    def stop_message_center(self):
        if self.message_event is not None:
            self.message_event.set()

        if self.listener_message_event is not None:
            self.listener_message_event.set()

    def check_message_stop_event(self):
        if self.message_event is not None and self.message_event.is_set():
            logging.info("Received message center stopping event.")
            raise MessageCenterStoppedException("Message center stopped (for sender)")

        if self.listener_message_event is not None and self.listener_message_event.is_set():
            logging.info("Received message center stopping event.")
            raise MessageCenterStoppedException("Message center stopped (for listener)")

    def send_message(self, topic, payload, run_id=None):
        message_entity = FedMLMessageEntity(topic=topic, payload=payload, run_id=run_id)
        self.sender_message_queue.put(message_entity.get_message_body())

    def send_message_json(self, topic, payload):
        self.send_message(topic, payload)

    def retry_sending_undelivered_message(self):
        for sender_message in self.sender_message_list:
            # Check if the message is published.
            message_record = FedMLMessageRecord(json_record=sender_message)
            is_published = False
            for published_message in self.published_message_ids:
                published_message_record = FedMLMessageRecord(json_record=published_message)
                if published_message_record.message_id == message_record.message_id:
                    is_published = True
                    break
            if is_published:
                continue

            # Retry to send the unpublished message based on the timeout value
            timeout_ms = time.time() * 1000.0 - message_record.timestamp
            if timeout_ms >= FedMLMessageCenter.PUBLISH_MESSAGE_RETRY_TIMEOUT and \
                self.retry_sending_count_map.get(message_record.message_id, 0) < \
                    FedMLMessageCenter.PUBLISH_MESSAGE_RETRY_COUNT:
                # Send the message
                message_entity = FedMLMessageEntity(message_body=message_record.message_body)
                message_id = self.sender_mqtt_mgr.send_message_json(message_entity.topic, message_entity.payload)
                self.retry_sending_count_map[message_record.message_id] += 1

                # Generate the new message record.
                sent_message_record = FedMLMessageRecord(message_id=message_id,
                                                         message_body=message_record.message_body)

                # Save the message
                self.save_message_record(message_entity.run_id, message_entity.device_id, sent_message_record)

    def run_sender(self, message_event, message_queue, message_center_name, process_name=None):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        if platform.system() != "Windows":
            os.setsid()

        self.message_event = message_event
        self.sender_message_queue = message_queue
        self.message_center_name = message_center_name
        self.setup_sender_mqtt_mgr()

        while True:
            message_entity = None
            message_body = None
            try:
                self.check_message_stop_event()
            except MessageCenterStoppedException as e:
                break

            # noinspection PyBroadException
            try:
                # Setup the mqtt connection
                self.setup_sender_mqtt_mgr()

                # Get the message from the queue
                try:
                    message_body = message_queue.get(block=False, timeout=0.1)
                except queue.Empty as e:  # If queue is empty, then break loop
                    message_body = None
                if message_body is None:
                    time.sleep(0.1)
                    # self.retry_sending_undelivered_message()
                    continue

                # Generate the message entity object
                message_entity = FedMLMessageEntity(message_body=message_body)

                # Send the message to mqtt server
                message_id = self.sender_mqtt_mgr.send_message_json(message_entity.topic, message_entity.payload)

                # Generate the message record.
                message_record = FedMLMessageRecord(message_id=message_id, message_body=message_body)

                # Cache the message
                self.cache_message_record(message_record, is_sender=True)

                # Save the message
                self.save_message_record(message_entity.run_id, message_entity.device_id, message_record)

            except Exception as e:
                if message_entity is not None:
                    logging.info(
                        f"Failed to send the message with topic {message_entity.topic}, "
                        f"payload {message_entity.payload}, {traceback.format_exc()}"
                    )
                else:
                    logging.info(f"Failed to send the message with body {message_body}, {traceback.format_exc()}")

        self.release_sender_mqtt_mgr()

    def get_protocol_communication_manager(self):
        return None

    def setup_listener_mqtt_mgr(self):
        if self.listener_mqtt_mgr is not None:
            return

        # self.listener_mqtt_mgr = self.get_protocol_communication_manager()
        # return

        self.listener_mqtt_mgr = MqttManager(
            self.listener_agent_config["mqtt_config"]["BROKER_HOST"],
            self.listener_agent_config["mqtt_config"]["BROKER_PORT"],
            self.listener_agent_config["mqtt_config"]["MQTT_USER"],
            self.listener_agent_config["mqtt_config"]["MQTT_PWD"],
            self.listener_agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_MessageCenter_@{}@{}".format(str(os.getpid()), str(uuid.uuid4()))
        )

        self.listener_mqtt_mgr.connect()
        self.listener_mqtt_mgr.loop_start()

    def get_listener_communication_manager(self):
        return self.listener_mqtt_mgr

    def release_listener_mqtt_mgr(self):
        #return
        try:
            if self.listener_mqtt_mgr is not None:
                self.listener_mqtt_mgr.loop_stop()
                self.listener_mqtt_mgr.disconnect()
                self.listener_mqtt_mgr = None
        except Exception as e:
            logging.error(
                f"Failed to release listener mqtt manager with Exception {e}. "
                f"Traceback: {traceback.format_exc()}"
            )
            pass

    def add_message_listener(self, topic, listener_func):
        if topic not in self.listener_topics:
            self.listener_topics.append(topic)
            self.listener_handler_funcs[topic] = listener_func.__name__

    def remove_message_listener(self, topic):
        if topic in self.listener_topics:
            self.listener_topics.remove(topic)
            self.listener_handler_funcs.pop(topic)

    def get_listener_handler(self, topic):
        return self.listener_handler_funcs.get(topic)

    def get_message_runner(self):
        return None

    def get_listener_message_queue(self):
        return self.listener_message_queue

    def setup_listener_message_queue(self):
        self.listener_message_queue = multiprocessing.Manager().Queue()

    def start_listener(
            self, sender_message_queue=None, listener_message_queue=None,
            sender_message_event=None, agent_config=None, message_center_name=None, extra_queues=None
    ):
        if self.listener_message_center_process is not None:
            return

        if listener_message_queue is None:
            if self.listener_message_queue is None:
                self.listener_message_queue = multiprocessing.Manager().Queue()
        else:
            self.listener_message_queue = listener_message_queue
        self.listener_message_event = multiprocessing.Event()
        self.listener_message_event.clear()
        self.listener_agent_config = agent_config
        message_runner = self
        message_runner.listener_agent_config = agent_config
        process_name = GeneralConstants.get_message_center_listener_process_name(message_center_name)
        if platform.system() == "Windows":
            self.listener_message_center_process = multiprocessing.Process(
                target=message_runner.run_listener_dispatcher, args=(
                    self.listener_message_event, self.listener_message_queue,
                    self.listener_handler_funcs, sender_message_queue,
                    sender_message_event, message_center_name, extra_queues, process_name
                )
            )
        else:
            self.listener_message_center_process = fedml.get_process(
                target=message_runner.run_listener_dispatcher, args=(
                    self.listener_message_event, self.listener_message_queue,
                    self.listener_handler_funcs, sender_message_queue,
                    sender_message_event, message_center_name, extra_queues, process_name
                )
            )
        self.listener_message_center_process.start()

    def check_listener_message_stop_event(self):
        if self.listener_message_event is not None and self.listener_message_event.is_set():
            logging.info("Received listener message center stopping event.")
            raise MessageCenterStoppedException("Message center stopped (for listener)")

    def listener_message_dispatch_center(self, topic, payload):
        self.receive_message_json(topic, payload)

    def listener_message_passthrough_dispatch_center(self, message):
        payload_obj = json.loads(message.payload)
        payload_obj["is_retain"] = message.retain
        payload = json.dumps(payload_obj)
        self.receive_message_json(message.topic, payload)

    def receive_message(self, topic, payload, run_id=None):
        message_entity = FedMLMessageEntity(topic=topic, payload=payload, run_id=run_id)
        self.listener_message_queue.put(message_entity.get_message_body())

    def receive_message_json(self, topic, payload):
        self.receive_message(topic, payload)

    def subscribe_msg(self, topic):
        self.listener_mqtt_mgr.add_message_listener(topic, self.listener_message_dispatch_center)
        self.listener_mqtt_mgr.subscribe_msg(topic)

    def unsubscribe_msg(self, topic):
        self.listener_mqtt_mgr.remove_message_listener(topic)
        self.listener_mqtt_mgr.unsubscribe_msg(topic)

    def run_listener_dispatcher(
            self, listener_message_event, listener_message_queue,
            listener_funcs, sender_message_queue, sender_message_event,
            message_center_name, extra_queues, process_name=None
    ):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        if platform.system() != "Windows":
            os.setsid()

        self.listener_message_event = listener_message_event
        self.listener_message_queue = listener_message_queue
        self.listener_handler_funcs = listener_funcs
        self.message_center_name = message_center_name
        self.sender_message_queue = sender_message_queue
        self.message_event = sender_message_event

        self.setup_listener_mqtt_mgr()

        if sender_message_queue is None:
            methodcaller(FedMLMessageCenter.FUNC_SETUP_MESSAGE_CENTER)(self)
        else:
            methodcaller(FedMLMessageCenter.FUNC_REBUILD_MESSAGE_CENTER, sender_message_queue)(self)

        if extra_queues is not None:
            methodcaller(FedMLMessageCenter.FUNC_PROCESS_EXTRA_QUEUES, extra_queues)(self)

        while True:
            message_entity = None
            try:
                self.check_listener_message_stop_event()
            except MessageCenterStoppedException as e:
                break

            # noinspection PyBroadException
            try:
                # Setup the mqtt connection
                self.setup_listener_mqtt_mgr()

                # Get the message from the queue
                try:
                    message_body = listener_message_queue.get(block=False, timeout=0.1)
                except queue.Empty as e:  # If queue is empty, then break loop
                    message_body = None
                if message_body is None:
                    time.sleep(0.1)
                    continue

                # Generate the message entity object
                message_entity = FedMLMessageEntity(message_body=message_body)

                # Generate the message record
                message_record = FedMLMessageRecord(message_id=str(uuid.uuid4()), message_body=message_body)

                # Cache the message
                self.cache_message_record(message_record, is_sender=False)

                # Save the message
                self.save_message_record(message_entity.run_id, message_entity.device_id,
                                         message_record, is_sender=False)

                # Dispatch the message to corresponding handler
                message_handler_func_name = self.listener_handler_funcs.get(message_entity.topic, None)
                if message_handler_func_name is not None:
                    methodcaller(message_handler_func_name, message_entity.topic, message_entity.payload)(self)
                else:
                    if hasattr(self, "callback_proxy_unknown_messages") and \
                            self.callback_proxy_unknown_messages is not None:
                        self.callback_proxy_unknown_messages(
                            message_entity.run_id, message_entity.topic, message_entity.payload)
            except Exception as e:
                if message_entity is not None:
                    logging.info(
                        f"Failed to dispatch messages with topic {message_entity.topic}, "
                        f"payload {message_entity.payload}, {traceback.format_exc()}")
                else:
                    logging.info(f"Failed to dispatch messages:  {traceback.format_exc()}")
        self.release_listener_mqtt_mgr()

    def cache_message_record(self, message_record, is_sender=True):
        # Save the message to the cached list.
        if is_sender:
            self.sender_message_list.append(message_record.get_json_record())
        else:
            self.receiver_message_list.append(message_record.get_json_record())

    def save_message_record(self, run_id, device_id, message_record, is_sender=True):
        # Check if we enable to log messages to file
        if not FedMLMessageCenter.ENABLE_SAVE_MESSAGE_TO_FILE:
            return

        # Log messages to file
        if is_sender:
            print(f"save sent message record: {message_record.get_json_record()}")
        else:
            print(f"save received message record: {message_record.get_json_record()}")
        saved_message_file = os.path.join(
            self.constants.message_log_dir,
            self.message_center_name,
            FedMLMessageCenter.MESSAGE_SENT_RECORDS_FILE if is_sender else
            FedMLMessageCenter.MESSAGE_RECEIVED_RECORDS_FILE
        )
        os.makedirs(os.path.dirname(saved_message_file), exist_ok=True)
        with open(saved_message_file, "a+") as f:
            f.writelines([json.dumps(message_record.get_json_record()) + "\n"])

    def save_published_message_record(self, message_id):
        # Check if we enable to log messages to file
        if not FedMLMessageCenter.ENABLE_SAVE_MESSAGE_TO_FILE:
            return

        # Log published message ids to file
        message_record = {"message_id": message_id, "timestamp": time.time_ns()/1000.0/1000.0}
        published_msg_record_file = os.path.join(
            self.constants.message_log_dir, self.message_center_name,
            FedMLMessageCenter.MESSAGE_SENT_SUCCESS_RECORDS_FILE)
        os.makedirs(os.path.dirname(published_msg_record_file), exist_ok=True)
        print(f"save sent success message record: {message_record}")
        with open(published_msg_record_file, "a+") as f:
            f.writelines([json.dumps(message_record) + "\n"])

    @staticmethod
    def rebuild_message_center_from_queue(sender_message_queue, listener_message_queue=None):
        message_center = FedMLMessageCenter(sender_message_queue=sender_message_queue,
                                            listener_message_queue=listener_message_queue)
        return message_center


class MessageCenterStoppedException(Exception):
    """ Message center stopped. """
    pass


class FedMLMessageCenterConstants:
    def __init__(self):
        global_services_dir = ClientConstants.get_global_services_dir()
        self.home_dir = expanduser("~")
        self.message_center_dir = os.path.join(global_services_dir, "message_center")
        self.message_log_dir = os.path.join(self.message_center_dir, "logs")
        os.makedirs(self.message_log_dir, exist_ok=True)
