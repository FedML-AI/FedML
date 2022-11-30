# -*-coding:utf-8-*-
import json
import logging
import traceback
import uuid
from typing import List
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
import paho.mqtt.client as mqtt
import yaml

from ..constants import CommunicationConstants
from ..mqtt.mqtt_manager import MqttManager
from ...distributed_storage.web3_storage.web3_storage import Web3Storage
from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from ..observer import Observer
from .....core.alg_frame.context import Context
import time


class MqttWeb3CommManager(BaseCommunicationManager):
    def __init__(
        self,
        config_path,
        web3_config_path,
        topic="fedml",
        client_rank=0,
        client_num=0,
        args=None
    ):
        self.broker_port = None
        self.broker_host = None
        self.mqtt_user = None
        self.mqtt_pwd = None
        self.keepalive_time = 180
        client_objects_str = str(args.client_id_list).replace('"', '"')
        client_objects_str = client_objects_str.replace("'", "")
        logging.info("origin client object " + str(args.client_id_list))
        logging.info("client object " + client_objects_str)
        self.client_id_list = json.loads(client_objects_str)

        self._topic = "fedml_" + str(topic) + "_"
        self.web3_storage = Web3Storage(web3_config_path)
        self.client_real_ids = []
        if args.client_id_list is not None:
            logging.info(
                "MqttWeb3CommManager args client_id_list: " + str(args.client_id_list)
            )
            self.client_real_ids = json.loads(args.client_id_list)

        self.group_server_id_list = None
        if hasattr(args, "group_server_id_list") and args.group_server_id_list is not None:
            self.group_server_id_list = args.group_server_id_list

        if args.rank == 0:
            if hasattr(args, "server_id"):
                self.edge_id = args.server_id
                self.server_id = args.server_id
            else:
                self.edge_id = 0
                self.server_id = 0
        else:
            if hasattr(args, "server_id"):
                self.server_id = args.server_id
            else:
                self.server_id = 0

            if hasattr(args, "client_id"):
                self.edge_id = args.client_id
            else:
                if len(self.client_real_ids) == 1:
                    self.edge_id = self.client_real_ids[0]
                else:
                    self.edge_id = 0

        self._observers: List[Observer] = []

        if client_rank is None:
            self._client_id = mqtt.base62(uuid.uuid4().int, padding=22)
        else:
            self._client_id = client_rank
        self.client_num = client_num
        logging.info("mqtt_web3.init: client_num = %d" % client_num)

        self.set_config_from_file(config_path)
        self.set_config_from_objects(config_path)

        self.client_active_list = dict()
        self.top_active_msg = CommunicationConstants.CLIENT_TOP_ACTIVE_MSG
        self.topic_last_will_msg = CommunicationConstants.CLIENT_TOP_LAST_WILL_MSG
        if args.rank == 0:
            self.top_active_msg = CommunicationConstants.SERVER_TOP_ACTIVE_MSG
            self.topic_last_will_msg = CommunicationConstants.SERVER_TOP_LAST_WILL_MSG
        self.last_will_msg = json.dumps({"ID": self.edge_id, "status": CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE})
        self.mqtt_mgr = MqttManager(self.broker_host, self.broker_port, self.mqtt_user, self.mqtt_pwd,
                                    self.keepalive_time,
                                    self._client_id, self.topic_last_will_msg,
                                    self.last_will_msg)
        self.mqtt_mgr.add_connected_listener(self.on_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_disconnected)
        self.mqtt_mgr.connect()

        self.is_connected = False

    @property
    def client_id(self):
        return self._client_id

    @property
    def topic(self):
        return self._topic

    def run_loop_forever(self):
        self.mqtt_mgr.loop_forever()

    def on_connected(self, mqtt_client_object):
        """
        [server]
        sending message topic (publish): serverID_clientID
        receiving message topic (subscribe): clientID

        [client]
        sending message topic (publish): clientID
        receiving message topic (subscribe): serverID_clientID

        """
        if self.is_connected:
            return
        self.mqtt_mgr.add_message_passthrough_listener(self._on_message)

        # Subscribe one topic
        if self.client_id == 0:
            # server
            self.subscribe_client_status_message()

            # logging.info("self.client_real_ids = {}".format(self.client_real_ids))
            for client_rank in range(0, self.client_num):
                real_topic = self._topic + str(self.client_real_ids[client_rank])
                result, mid = mqtt_client_object.subscribe(real_topic, qos=2)

                # logging.info(
                #     "mqtt_web3.on_connect: subscribes real_topic = %s, mid = %s, result = %s"
                #     % (real_topic, mid, str(result))
                # )
            # logging.info("mqtt_web3.on_connect: server subscribes")
            self._notify_connection_ready()
        else:
            # client
            real_topic = self._topic + str(self.server_id) + "_" + str(self.client_real_ids[0])
            result, mid = mqtt_client_object.subscribe(real_topic, qos=2)

            self._notify_connection_ready()

            # logging.info(
            #     "mqtt_web3.on_connect: client subscribes real_topic = %s, mid = %s, result = %s"
            #     % (real_topic, mid, str(result))
            # )
        self.is_connected = True

    def on_disconnected(self, mqtt_client_object):
        self.is_connected = False

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _notify_connection_ready(self):
        msg_params = Message()
        msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _notify(self, msg_obj):
        msg_params = Message()
        msg_params.init_from_json_object(msg_obj)
        msg_type = msg_params.get_type()
        logging.info("mqtt_web3.notify: msg type = %s" % msg_type)
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _on_message_impl(self, msg):
        json_payload = str(msg.payload, encoding="utf-8")
        payload_obj = json.loads(json_payload)
        sender_id = payload_obj.get(Message.MSG_ARG_KEY_SENDER, "")
        receiver_id = payload_obj.get(Message.MSG_ARG_KEY_RECEIVER, "")
        web3_key_str = payload_obj.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
        web3_key_str = str(web3_key_str).strip(" ")

        if web3_key_str != "":
            logging.info(
                "mqtt_web3.on_message: use web3 pack, web3 message key %s" % web3_key_str
            )

            model_params = self.web3_storage.read_model(web3_key_str)
            Context().add("received_model_cid", web3_key_str)
            logging.info("Received model cid {}".format(Context().get("received_model_cid")))

            logging.info(
                "mqtt_web3.on_message: model params length %d" % len(model_params)
            )

            # replace the web3 object key with raw model params
            payload_obj[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params
        else:
            logging.info("mqtt_web3.on_message: not use web3 pack")

        self._notify(payload_obj)

    def _on_message(self, msg):
        try:
            self._on_message_impl(msg)
        except Exception as e:
            logging.error("mqtt_web3.on_message exception: {}".format(traceback.format_exc()))

    def send_message(self, msg: Message):
        """
        [server]
        sending message topic (publish): fedml_runid_serverID_clientID
        receiving message topic (subscribe): fedml_runid_clientID

        [client]
        sending message topic (publish): fedml_runid_clientID
        receiving message topic (subscribe): fedml_runid_serverID_clientID

        """
        sender_id = msg.get_sender_id()
        receiver_id = msg.get_receiver_id()
        if self.client_id == 0:
            # topic = "fedml" + "_" + "run_id" + "_0" + "_" + "client_id"
            topic = self._topic + str(self.server_id) + "_" + str(receiver_id)
            logging.info("mqtt_web3.send_message: msg topic = %s" % str(topic))

            payload = msg.get_params()
            model_params_obj = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
            if model_params_obj != "":
                # web3
                logging.info("mqtt_web3.send_message: to python client.")
                message_key = model_url = self.web3_storage.write_model(model_params_obj)
                Context().add("sent_model_cid", model_url)
                logging.info("Sent model cid {}".format(Context().get("sent_model_cid")))
                logging.info(
                    "mqtt_web3.send_message: web3+MQTT msg sent, web3 message key = %s"
                    % message_key
                )
                model_params_key_url = {
                    "key": message_key,
                    "url": model_url,
                    "obj": model_params_obj,
                }
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params_key_url["key"]
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL] = model_params_key_url[
                    "url"
                ]
                self.mqtt_mgr.send_message(topic, json.dumps(payload))
            else:
                # pure MQTT
                logging.info("mqtt_web3.send_message: MQTT msg sent")
                self.mqtt_mgr.send_message(topic, json.dumps(payload))

        else:
            # client
            topic = self._topic + str(msg.get_sender_id())

            payload = msg.get_params()
            model_params_obj = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
            if model_params_obj != "":
                # web3
                message_key = model_url = self.web3_storage.write_model(model_params_obj)
                Context().add("sent_model_cid", model_url)
                logging.info("Sent model cid {}".format(Context().get("sent_model_cid")))
                logging.info(
                    "mqtt_web3.send_message: web3+MQTT msg sent, message_key = %s"
                    % message_key
                )
                model_params_key_url = {
                    "key": message_key,
                    "url": model_url,
                    "obj": model_params_obj,
                }
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params_key_url["key"]
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL] = model_params_key_url[
                    "url"
                ]
                self.mqtt_mgr.send_message(topic, json.dumps(payload))
            else:
                logging.info("mqtt_web3.send_message: MQTT msg sent")
                self.mqtt_mgr.send_message(topic, json.dumps(payload))

    def send_message_json(self, topic_name, json_message):
        self.mqtt_mgr.send_message_json(topic_name, json_message)

    def handle_receive_message(self):
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        self.run_loop_forever()
        MLOpsProfilerEvent.log_to_wandb({"TotalTime": time.time() - start_listening_time})

    def stop_receive_message(self):
        logging.info("mqtt_web3.stop_receive_message: stopping...")
        self.mqtt_mgr.loop_stop()
        self.mqtt_mgr.disconnect()

    def set_config_from_file(self, config_file_path):
        try:
            with open(config_file_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.broker_host = config["BROKER_HOST"]
                self.broker_port = config["BROKER_PORT"]
                self.mqtt_user = None
                self.mqtt_pwd = None
                if "MQTT_USER" in config:
                    self.mqtt_user = config["MQTT_USER"]
                if "MQTT_PWD" in config:
                    self.mqtt_pwd = config["MQTT_PWD"]
        except Exception as e:
            pass

    def set_config_from_objects(self, mqtt_config):
        self.broker_host = mqtt_config["BROKER_HOST"]
        self.broker_port = mqtt_config["BROKER_PORT"]
        self.mqtt_user = None
        self.mqtt_pwd = None
        if "MQTT_USER" in mqtt_config:
            self.mqtt_user = mqtt_config["MQTT_USER"]
        if "MQTT_PWD" in mqtt_config:
            self.mqtt_pwd = mqtt_config["MQTT_PWD"]

    def callback_client_last_will_msg(self, topic, payload):
        msg = json.loads(payload)
        edge_id = msg.get("ID", None)
        status = msg.get("status", CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE)
        if edge_id is not None and status == CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE:
            if self.client_active_list.get(edge_id, None) is not None:
                self.client_active_list.pop(edge_id)

    def callback_client_active_msg(self, topic, payload):
        msg = json.loads(payload)
        edge_id = msg.get("ID", None)
        status = msg.get("status", CommunicationConstants.MSG_CLIENT_STATUS_IDLE)
        if edge_id is not None:
            self.client_active_list[edge_id] = status

    def subscribe_client_status_message(self):
        # Setup MQTT message listener to the last will message form the client.
        self.mqtt_mgr.add_message_listener(CommunicationConstants.CLIENT_TOP_LAST_WILL_MSG,
                                           self.callback_client_last_will_msg)

        # Setup MQTT message listener to the active status message from the client.
        self.mqtt_mgr.add_message_listener(CommunicationConstants.CLIENT_TOP_ACTIVE_MSG,
                                           self.callback_client_active_msg)

    def get_client_status(self, client_id):
        return self.client_active_list.get(client_id, CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE)

    def get_client_list_status(self):
        return self.client_active_list


if __name__ == "__main__":
    pass