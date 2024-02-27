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


class FedMLMessageCenter:

    def __init__(self, agent_config=None, message_queue=None):
        self.agent_config = agent_config
        self.message_queue = message_queue
        self.message_event = None
        self.message_center_process = None
        self.client_mqtt_mgr = None
        self.mlops_metrics = None
        self.client_mqtt_lock = None
        self.client_mqtt_is_connected = False

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def on_client_mqtt_disconnected(self, mqtt_client_object):
        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock.release()

    def on_client_mqtt_connected(self, mqtt_client_object):
        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()

        self.mlops_metrics.set_messenger(self)

        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = True
        self.client_mqtt_lock.release()

    def setup_client_mqtt_mgr(self):
        if self.client_mqtt_mgr is not None:
            return

        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_mgr = MqttManager(
            self.agent_config["mqtt_config"]["BROKER_HOST"],
            self.agent_config["mqtt_config"]["BROKER_PORT"],
            self.agent_config["mqtt_config"]["MQTT_USER"],
            self.agent_config["mqtt_config"]["MQTT_PWD"],
            self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_MessageCenter_@{}@{}".format(str(os.getpid()), str(uuid.uuid4()))
        )

        self.client_mqtt_mgr.add_connected_listener(self.on_client_mqtt_connected)
        self.client_mqtt_mgr.add_disconnected_listener(self.on_client_mqtt_disconnected)
        self.client_mqtt_mgr.connect()
        self.client_mqtt_mgr.loop_start()

        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self)

    def release_client_mqtt_mgr(self):
        try:
            if self.client_mqtt_mgr is not None:
                self.client_mqtt_mgr.loop_stop()
                self.client_mqtt_mgr.disconnect()

            self.client_mqtt_lock.acquire()
            if self.client_mqtt_mgr is not None:
                self.client_mqtt_is_connected = False
                self.client_mqtt_mgr = None
            self.client_mqtt_lock.release()
        except Exception as e:
            logging.error(
                f"Failed to release client mqtt manager with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

    def get_message_queue(self):
        return self.message_queue

    def start(self):
        self.message_queue = Queue()
        self.message_event = multiprocessing.Event()
        self.message_event.clear()
        message_center = FedMLMessageCenter(self.agent_config, message_queue=self.message_queue)
        self.message_center_process = Process(
            target=message_center.run, args=(
                self.message_event, self.message_queue,
            )
        )
        self.message_center_process.start()

    def stop(self):
        if self.message_event is not None:
            self.message_event.set()

    def check_message_stop_event(self):
        if self.message_event is not None and self.message_event.is_set():
            logging.info("Received message center stopping event.")
            raise Exception("Message center stopped")

    def send_message(self, topic, payload, run_id=None):
        message_entry = FedMLMessageEntry(topic=topic, payload=payload, run_id=run_id)
        self.message_queue.put(message_entry.get_message_body())

    def send_message_json(self, topic, payload):
        self.send_message(topic, payload)

    def run(self, message_event, message_queue):
        self.message_event = message_event
        self.message_queue = message_queue
        self.setup_client_mqtt_mgr()
        time.sleep(5)

        while True:
            try:
                self.check_message_stop_event()
            except MessageCenterStoppedException as e:
                break

            try:
                self.setup_client_mqtt_mgr()

                try:
                    message_body = self.message_queue.get(block=False, timeout=0.1)
                except queue.Empty as e:  # If queue is empty, then break loop
                    message_body = None
                if message_body is None:
                    time.sleep(0.1)
                    continue

                message_entry = FedMLMessageEntry(message_body=message_body)
                self.client_mqtt_mgr.send_message_json(message_entry.topic, message_entry.payload)
            except Exception as e:
                logging.info(
                    f"Failed to send the message with topic {message_entry.topic}, payload {message_entry.payload}")

        self.release_client_mqtt_mgr()


class FedMLMessageEntry(object):
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
