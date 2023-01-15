# -*-coding:utf-8-*-
import json
import logging
import traceback
import uuid
from typing import List
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
import yaml

from fedml.model.linear.lr import LogisticRegression
from fedml.model.cv.cnn import CNN_WEB
from ..constants import CommunicationConstants
from ..mqtt.mqtt_manager import MqttManager
from ..s3.remote_storage import S3Storage
from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from ..observer import Observer
import time


class MqttS3MultiClientsCommManager(BaseCommunicationManager):
    MESSAGE_CACHE_SENT_JSON_TOPIC = "message_json_topic"
    MESSAGE_CACHE_SENT_JSON_PAYLOAD = "message_json_payload"

    def __init__(
            self,
            config_path,
            s3_config_path,
            topic="fedml",
            client_rank=0,
            client_num=0,
            args=None
    ):
        self.args = args
        self.broker_port = None
        self.broker_host = None
        self.mqtt_user = None
        self.mqtt_pwd = None
        self.keepalive_time = 10
        client_objects_str = str(args.client_id_list).replace('"', '"')
        client_objects_str = client_objects_str.replace("'", "")
        self.isBrowser = False
        if hasattr(args, "is_browser"):
            self.isBrowser = args.is_browser
        logging.info(args.__dict__)
        self.dataSetType = args.dataset
        logging.info("is browser device: " + str(self.isBrowser))
        logging.info("origin client object " + str(args.client_id_list))
        logging.info("client object " + client_objects_str)
        self.client_id_list = json.loads(client_objects_str)

        self._topic = "fedml_" + str(topic) + "_"
        self.s3_storage = S3Storage(s3_config_path)
        self.client_real_ids = []
        if args.client_id_list is not None:
            logging.info(
                "MqttS3CommManager args client_id_list: " + str(args.client_id_list)
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

        self._client_id = "FedML_CS_{}_{}".format(str(args.run_id), str(self.edge_id))
        self.client_num = client_num
        logging.info("mqtt_s3.init: client_num = %d" % client_num)

        self.set_config_from_file(config_path)
        self.set_config_from_objects(config_path)

        self.client_active_list = dict()
        self.top_active_msg = CommunicationConstants.CLIENT_TOP_ACTIVE_MSG
        self.topic_last_will_msg = CommunicationConstants.CLIENT_TOP_LAST_WILL_MSG
        if args.rank == 0:
            self.top_active_msg = CommunicationConstants.SERVER_TOP_ACTIVE_MSG
            self.topic_last_will_msg = CommunicationConstants.SERVER_TOP_LAST_WILL_MSG
        self.last_will_msg = json.dumps({"ID": self.edge_id, "status": CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE})
        self.mqtt_mgr = MqttManager(
            config_path["BROKER_HOST"],
            config_path["BROKER_PORT"],
            config_path["MQTT_USER"],
            config_path["MQTT_PWD"],
            config_path["MQTT_KEEPALIVE"],
            self._client_id,
            last_will_topic=self.topic_last_will_msg,
            last_will_msg=self.last_will_msg
        )
        self.mqtt_mgr.add_connected_listener(self.on_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_disconnected)
        self.mqtt_mgr.connect()

        self.is_connected = False
        self.sent_msg_caches: List[Message] = []

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
        self.mqtt_mgr.add_message_passthrough_listener(self._on_message)

        # Subscribe one topic
        if self.args.rank == 0:
            # server
            self.subscribe_client_status_message()

            # logging.info("self.client_real_ids = {}".format(self.client_real_ids))
            for client_rank in range(0, self.client_num):
                real_topic = self._topic + str(self.client_real_ids[client_rank])
                result, mid = mqtt_client_object.subscribe(real_topic, qos=2)

                # logging.info(
                #     "mqtt_s3.on_connect: subscribes real_topic = %s, mid = %s, result = %s"
                #     % (real_topic, mid, str(result))
                # )
            # logging.info("mqtt_s3.on_connect: server subscribes")

            self.send_cache_msgs()
            self._notify_connection_ready()
        else:
            # client
            real_topic = self._topic + str(self.server_id) + "_" + str(self.client_real_ids[0])
            result, mid = mqtt_client_object.subscribe(real_topic, qos=2)

            self.send_cache_msgs()
            self._notify_connection_ready()

            # logging.info(
            #     "mqtt_s3.on_connect: client subscribes real_topic = %s, mid = %s, result = %s"
            #     % (real_topic, mid, str(result))
            # )
        self.is_connected = True

    def send_cache_msgs(self):
        pass
        # sent_msgs: List[Message] = []
        # for msg in self.sent_msg_caches:
        #     sent_json_topic = msg.get(MqttS3MultiClientsCommManager.MESSAGE_CACHE_SENT_JSON_TOPIC)
        #     sent_json_payload = msg.get(MqttS3MultiClientsCommManager.MESSAGE_CACHE_SENT_JSON_PAYLOAD)
        #     if sent_json_topic is None:
        #         print("send cache obj")
        #         sent = self.send_message(msg, wait_for_publish=False, not_cache=True)
        #     else:
        #         print("send cache json")
        #         sent = self.send_message_json(sent_json_topic, sent_json_payload)
        #     # if sent:
        #     sent_msgs.append(msg)
        #
        # # for msg in sent_msgs:
        # #     del self.sent_msg_caches[msg]
        # if len(sent_msgs) > 0:
        #     sent_msgs.clear()
        # self.sent_msg_caches.clear()

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
        logging.info("mqtt_s3.notify: msg type = %s" % msg_type)
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _on_message_impl(self, msg):
        json_payload = str(msg.payload, encoding="utf-8")
        payload_obj = json.loads(json_payload)
        logging.info(
            "mqtt_s3 receive msg %s" % payload_obj
        )
        sender_id = payload_obj.get(Message.MSG_ARG_KEY_SENDER, "")
        receiver_id = payload_obj.get(Message.MSG_ARG_KEY_RECEIVER, "")
        s3_key_str = payload_obj.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
        s3_key_str = str(s3_key_str).strip(" ")
        device = payload_obj.get("deviceType", "")
        logging.info(
            "mqtt_s3 receive msg deviceType %s" % device
        )

        if s3_key_str != "":
            logging.info(
                "mqtt_s3.on_message: use s3 pack, s3 message key %s" % s3_key_str
            )

            # model_params = self.s3_storage.read_model(s3_key_str)
            # read model from client
            if device == 'web':
                # init model structure from client
                if self.dataSetType == 'mnist':
                    py_model = LogisticRegression(28 * 28, 10)
                elif self.dataSetType == 'cifar10':
                    py_model = CNN_WEB()

                model_params = self.s3_storage.read_model_web(s3_key_str, py_model)
            else:
                model_params = self.s3_storage.read_model(s3_key_str)

            logging.info(
                "mqtt_s3.on_message: model params length %d" % len(model_params)
            )

            model_url = payload_obj.get(Message.MSG_ARG_KEY_MODEL_PARAMS_URL, "")
            logging.info("mqtt_s3.on_message: model url {}".format(model_url))

            # replace the S3 object key with raw model params
            payload_obj[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params
            payload_obj[Message.MSG_ARG_KEY_MODEL_PARAMS_KEY] = s3_key_str
        else:
            logging.info("mqtt_s3.on_message: not use s3 pack")

        self._notify(payload_obj)

    def _on_message(self, msg):
        try:
            self._on_message_impl(msg)
        except Exception as e:
            logging.error("mqtt_s3.on_message exception: {}".format(traceback.format_exc()))

    def send_message(self, msg: Message, wait_for_publish=False, not_cache=False):
        """
        [server]
        sending message topic (publish): fedml_runid_serverID_clientID
        receiving message topic (subscribe): fedml_runid_clientID

        [client]
        sending message topic (publish): fedml_runid_clientID
        receiving message topic (subscribe): fedml_runid_serverID_clientID

        """
        # if self.mqtt_mgr.is_connected() is False:
        #     if not not_cache:
        #         self.sent_msg_caches.append(msg)
        #     return False
        # print("msg cache num {}.".format(str(len(self.sent_msg_caches))))

        sent_result = None
        sender_id = msg.get_sender_id()
        receiver_id = msg.get_receiver_id()
        if self.args.rank == 0:
            # topic = "fedml" + "_" + "run_id" + "_0" + "_" + "client_id"
            topic = self._topic + str(self.server_id) + "_" + str(receiver_id)
            logging.info("mqtt_s3.send_message: msg topic = %s" % str(topic))

            payload = msg.get_params()
            model_params_obj = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
            model_url = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS_URL, "")
            model_key = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS_KEY, "")
            if model_params_obj != "":
                # S3
                if model_url == "":
                    model_key = topic + "_" + str(uuid.uuid4())
                    if self.isBrowser:
                        model_url = self.s3_storage.write_model_web(model_key, model_params_obj)
                    else:
                        model_url = self.s3_storage.write_model(model_key, model_params_obj)

                logging.info(
                    "mqtt_s3.send_message: S3+MQTT msg sent, s3 message key = %s"
                    % model_key
                )
                logging.info("mqtt_s3.send_message: to python client.")

                payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_key
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL] = model_url
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS_KEY] = model_key
                sent_result = self.mqtt_mgr.send_message(topic, json.dumps(payload))
            else:
                # pure MQTT
                sent_result = self.mqtt_mgr.send_message(topic, json.dumps(payload))
        else:
            # client
            topic = self._topic + str(msg.get_sender_id())
            message_key = topic + "_" + str(uuid.uuid4())

            payload = msg.get_params()
            model_params_obj = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
            if model_params_obj != "":
                # S3
                logging.info(
                    "mqtt_s3.send_message: S3+MQTT msg sent, message_key = %s"
                    % message_key
                )
                model_url = self.s3_storage.write_model(message_key, model_params_obj)
                model_params_key_url = {
                    "key": message_key,
                    "url": model_url,
                    "obj": model_params_obj,
                }
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params_key_url["key"]
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL] = model_params_key_url[
                    "url"
                ]
                logging.info(
                    "mqtt_s3.send_message: client s3, topic = %s"
                    % topic
                )
                sent_result = self.mqtt_mgr.send_message(topic, json.dumps(payload))
            else:
                logging.info("mqtt_s3.send_message: MQTT msg sent")
                sent_result = self.mqtt_mgr.send_message(topic, json.dumps(payload))

        if sent_result is not None and not sent_result:
            # if not not_cache:
            #     self.sent_msg_caches.append(msg)
            return False

        return True

    def send_message_json(self, topic_name, json_message):
        sent_result = self.mqtt_mgr.send_message_json(topic_name, json_message)
        if sent_result is not None and not sent_result:
            # msb_obj = Message()
            # msb_obj.add(MqttS3MultiClientsCommManager.MESSAGE_CACHE_SENT_JSON_TOPIC, topic_name)
            # msb_obj.add(MqttS3MultiClientsCommManager.MESSAGE_CACHE_SENT_JSON_PAYLOAD, json_message)
            # self.sent_msg_caches.append(msb_obj)
            return False

        return True

    def handle_receive_message(self):
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        self.run_loop_forever()
        MLOpsProfilerEvent.log_to_wandb({"TotalTime": time.time() - start_listening_time})

    def stop_receive_message(self):
        logging.info("mqtt_s3.stop_receive_message: stopping...")
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
        self.mqtt_mgr.add_message_listener(self.topic_last_will_msg,
                                           self.callback_client_last_will_msg)

        # Setup MQTT message listener to the active status message from the client.
        self.mqtt_mgr.add_message_listener(self.top_active_msg,
                                           self.callback_client_active_msg)

    def get_client_status(self, client_id):
        return self.client_active_list.get(client_id, CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE)

    def get_client_list_status(self):
        return self.client_active_list
