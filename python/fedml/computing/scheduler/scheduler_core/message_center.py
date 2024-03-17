import logging
import os
import threading
import time
import traceback
import uuid
import multiprocessing
from multiprocessing import Process, Queue
import queue

from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ....core.mlops.mlops_metrics import MLOpsMetrics
from operator import methodcaller


class FedMLMessageCenter:
    FUNC_SETUP_MESSAGE_CENTER = "setup_message_center"
    FUNC_REBUILD_MESSAGE_CENTER = "rebuild_message_center"

    def __init__(self, agent_config=None, message_queue=None, listener_message_queue=None):
        self.sender_agent_config = agent_config
        self.listener_agent_config = agent_config
        self.message_queue = message_queue
        self.message_event = None
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
        self.listener_message_queue = None
        self.listener_message_event = None
        self.listener_message_center_process = None

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
        self.sender_mqtt_mgr.connect()
        self.sender_mqtt_mgr.loop_start()

        if self.sender_mlops_metrics is None:
            self.sender_mlops_metrics = MLOpsMetrics()
        self.sender_mlops_metrics.set_messenger(self)

    def release_sender_mqtt_mgr(self):
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

    def get_message_queue(self):
        return self.message_queue

    def start_sender(self):
        self.message_queue = Queue()
        self.message_event = multiprocessing.Event()
        self.message_event.clear()
        message_center = FedMLMessageCenter(agent_config=self.sender_agent_config, message_queue=self.message_queue)
        self.message_center_process = Process(
            target=message_center.run_sender, args=(
                self.message_event, self.message_queue,
            )
        )
        self.message_center_process.start()

    def stop(self):
        if self.message_event is not None:
            self.message_event.set()

        if self.listener_message_event is not None:
            self.listener_message_event.set()

    def check_message_stop_event(self):
        if self.message_event is not None and self.message_event.is_set():
            logging.info("Received message center stopping event.")
            raise MessageCenterStoppedException("Message center stopped (for sender)")

    def send_message(self, topic, payload, run_id=None):
        message_entity = FedMLMessageEntity(topic=topic, payload=payload, run_id=run_id)
        self.message_queue.put(message_entity.get_message_body())

    def send_message_json(self, topic, payload):
        self.send_message(topic, payload)

    def run_sender(self, message_event, message_queue):
        self.message_event = message_event
        self.message_queue = message_queue
        self.setup_sender_mqtt_mgr()
        time.sleep(5)

        while True:
            try:
                self.check_message_stop_event()
            except MessageCenterStoppedException as e:
                break

            try:
                self.setup_sender_mqtt_mgr()

                try:
                    message_body = self.message_queue.get(block=False, timeout=0.1)
                except queue.Empty as e:  # If queue is empty, then break loop
                    message_body = None
                if message_body is None:
                    time.sleep(0.1)
                    continue

                message_entity = FedMLMessageEntity(message_body=message_body)
                self.sender_mqtt_mgr.send_message_json(message_entity.topic, message_entity.payload)
            except Exception as e:
                logging.info(
                    f"Failed to send the message with topic {message_entity.topic}, payload {message_entity.payload}, {traceback.format_exc()}")

        self.release_sender_mqtt_mgr()

    def setup_listener_mqtt_mgr(self):
        if self.listener_mqtt_mgr is not None:
            return

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

    def release_listener_mqtt_mgr(self):
        try:
            if self.listener_mqtt_mgr is not None:
                self.listener_mqtt_mgr.loop_stop()
                self.listener_mqtt_mgr.disconnect()
                self.listener_mqtt_mgr = None
        except Exception as e:
            logging.error(
                f"Failed to release listener mqtt manager with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

    def add_message_listener(self, topic, listener_func):
        if topic not in self.listener_topics:
            self.listener_topics.append(topic)
            self.listener_handler_funcs[topic] = listener_func.__name__

    def remove_message_listener(self, topic):
        if topic in self.listener_topics:
            self.listener_topics.remove(topic)
            self.listener_handler_funcs.pop(topic)

    def get_runner(self):
        return None

    def start_listener(self, sender_message_queue=None, agent_config=None):
        if self.listener_message_center_process is not None:
            return

        self.listener_message_queue = Queue()
        self.listener_message_event = multiprocessing.Event()
        self.listener_message_event.clear()
        self.listener_agent_config = agent_config
        message_runner = self.get_runner()
        message_runner.listener_agent_config = agent_config
        self.listener_message_center_process = Process(
            target=message_runner.run_listener_dispatcher, args=(
                self.listener_message_event, self.listener_message_queue,
                self.listener_handler_funcs, sender_message_queue
            )
        )
        self.listener_message_center_process.start()

    def check_listener_message_stop_event(self):
        if self.listener_message_event is not None and self.listener_message_event.is_set():
            logging.info("Received listener message center stopping event.")
            raise MessageCenterStoppedException("Message center stopped (for listener)")

    def listener_message_dispatch_center(self, topic, payload):
        self.receive_message_json(topic, payload)

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
            self, message_event, message_queue, listener_funcs, sender_message_queue):
        self.listener_message_event = message_event
        self.listener_message_queue = message_queue
        self.listener_handler_funcs = listener_funcs

        self.setup_listener_mqtt_mgr()

        if sender_message_queue is None:
            methodcaller(FedMLMessageCenter.FUNC_SETUP_MESSAGE_CENTER)(self)
        else:
            methodcaller(FedMLMessageCenter.FUNC_REBUILD_MESSAGE_CENTER, sender_message_queue)(self)

        while True:
            try:
                self.check_listener_message_stop_event()
            except MessageCenterStoppedException as e:
                break

            try:
                self.setup_listener_mqtt_mgr()

                try:
                    message_body = self.listener_message_queue.get(block=False, timeout=0.1)
                except queue.Empty as e:  # If queue is empty, then break loop
                    message_body = None
                if message_body is None:
                    time.sleep(0.1)
                    continue

                message_entity = FedMLMessageEntity(message_body=message_body)

                message_handler_func_name = self.listener_handler_funcs.get(message_entity.topic, None)
                if message_handler_func_name is not None:
                    methodcaller(message_handler_func_name, message_entity.topic, message_entity.payload)(self)
            except Exception as e:
                logging.info(
                    f"Failed to dispatch messages with topic {message_entity.topic}, payload {message_entity.payload}, {traceback.format_exc()}")

        self.release_listener_mqtt_mgr()

class FedMLMessageEntity(object):
    def __init__(self, topic=None, payload=None, run_id=None, message_body: dict = None):
        self.topic = topic
        self.payload = payload
        self.run_id = run_id
        if message_body is not None:
            self.from_message_body(message_body=message_body)

    def from_message_body(self, message_body: dict = None):
        self.topic = message_body.get("topic", None)
        self.payload = message_body.get("payload", None)
        self.run_id = message_body.get("run_id", None)

    def get_message_body(self):
        message_body = {"topic": self.topic, "payload": self.payload, "run_id": self.run_id}
        return message_body


class MessageCenterStoppedException(Exception):
    """ Message center stopped. """
    pass
