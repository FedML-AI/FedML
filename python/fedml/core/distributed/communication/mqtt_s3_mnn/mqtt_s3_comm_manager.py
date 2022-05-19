# -*-coding:utf-8-*-
import json
import time
import traceback
import uuid
from typing import List

import paho.mqtt.client as mqtt
import yaml

from .remote_storage import S3Storage
from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from ..observer import Observer
import logging


class MqttS3MNNCommManager(BaseCommunicationManager):
    def __init__(
        self,
        config_path,
        s3_config_path,
        topic="fedml",
        client_id=0,
        client_num=0,
        args=None,
        bind_port=0,
    ):
        self.args = args
        self._topic = "fedml_" + str(topic) + "_"
        self.s3_storage = S3Storage(s3_config_path)
        self.client_real_ids = []
        logging.info(
            "MqttS3CommManager args client_id_list: " + str(args.client_id_list)
        )
        if args is not None:
            self.client_real_ids = json.loads(args.client_id_list)

        self.model_params_key_map = list()

        self._unacked_sub = list()
        self._observers: List[Observer] = []
        if client_id is None:
            self._client_id = mqtt.base62(uuid.uuid4().int, padding=22)
        else:
            self._client_id = client_id
        self.client_num = client_num
        logging.info("mqtt_s3.init: client_num = %d" % client_num)

        self.set_config_from_file(config_path)
        self.set_config_from_objects(config_path)
        # Construct a Client
        self.mqtt_connection_id = mqtt.base62(uuid.uuid4().int, padding=22)
        self._client = mqtt.Client(
            client_id=str(self.mqtt_connection_id), clean_session=True
        )
        self._client.enable_logger()

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        # self._client.on_log = self.on_log

        self._client.username_pw_set(self.mqtt_user, self.mqtt_pwd)
        _will_msg = {"ID": f"{self.mqtt_connection_id}", "stat": "Online"}
        self._client.will_set(
            "W/topic", payload=json.dumps(_will_msg), qos=0, retain=True
        )

        logging.info(
            "mqtt_s3.init: connecting to MQTT server(local port %d..." % bind_port
        )
        self._client.connect(
            self.broker_host, self.broker_port, 180, bind_port=bind_port
        )

    def on_log(self, mqttc, obj, level, string):
        logging.info("mqtt_s3.on_log: " + string)

    def run_loop_forever(self):
        self._client.loop_forever()

    def __del__(self):
        self._client.loop_stop()
        self._client.disconnect()

    @property
    def client_id(self):
        return self._client_id

    @property
    def topic(self):
        return self._topic

    def _on_connect_impl(self, client, userdata, flags, rc):
        """
        [server]
        sending message topic (publish): serverID_clientID
        receiving message topic (subscribe): clientID

        [client]
        sending message topic (publish): clientID
        receiving message topic (subscribe): serverID_clientID

        """
        logging.info(
            "mqtt_s3.on_connect: connection returned with result code:" + str(rc)
        )
        # subscribe one topic
        if self.client_id == 0:
            # server
            for client_ID in range(1, self.client_num + 1):
                real_topic = self._topic + str(self.client_real_ids[client_ID - 1])
                result, mid = client.subscribe(real_topic, 0)

                self._unacked_sub.append(mid)
                logging.info(
                    "mqtt_s3.on_connect: server subscribes real_topic = %s, mid = %s, result = %s"
                    % (real_topic, mid, str(result))
                )
        else:
            # client
            real_topic = self._topic + str(0) + "_" + str(self.client_real_ids[0])
            result, mid = client.subscribe(real_topic, 0)
            self._unacked_sub.append(mid)

            logging.info(
                "mqtt_s3.on_connect: client subscribes real_topic = %s, mid = %s, result = %s"
                % (real_topic, mid, str(result))
            )

    def _on_connect(self, client, userdata, flags, rc):
        try:
            self._on_connect_impl(client, userdata, flags, rc)
        except:
            traceback.print_exc()
            quit(0)

    @staticmethod
    def _on_disconnect(client, userdata, rc):
        logging.info(
            "mqtt_s3.on_disconnect: disconnection returned result %s, user data %s"
            % (str(rc), str(userdata))
        )

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        logging.info("mqtt_s3.onSubscribe: mid = %s" % str(mid))
        self._unacked_sub.remove(mid)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _notify(self, msg_obj):
        msg_params = Message()
        msg_params.init_from_json_object(msg_obj)
        msg_type = msg_params.get_type()
        logging.info("mqtt_s3.notify: msg type = %d" % msg_type)
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _on_message_impl(self, client, userdata, msg):
        logging.info("--------------------------")
        json_payload = str(msg.payload, encoding="utf-8")
        payload_obj = json.loads(json_payload)
        logging.info("mqtt_s3.on_message: payload_obj %s" % payload_obj)
        s3_key_str = payload_obj.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
        s3_key_str = str(s3_key_str).strip(" ")
        if s3_key_str != "":
            logging.info(
                "mqtt_s3.on_message: use s3 pack, s3 message key %s" % s3_key_str
            )
            model_params = self.args.model_file_cache_folder + "/" + s3_key_str
            self.s3_storage.download_model_file(s3_key_str, model_params)

            logging.info(
                "mqtt_s3.on_message: model params length %d" % len(model_params)
            )

            # replace the S3 object key with raw model params
            payload_obj[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params
        else:
            logging.info("mqtt_s3.on_message: not use s3 pack")

        self._notify(payload_obj)

    def _on_message(self, client, userdata, msg):
        try:
            self._on_message_impl(client, userdata, msg)
        except:
            traceback.print_exc()
            quit(0)

    def send_message(self, msg: Message):
        """
        [server]
        sending message topic (publish): fedml_runid_serverID_clientID
        receiving message topic (subscribe): fedml_runid_clientID

        [client]
        sending message topic (publish): fedml_runid_clientID
        receiving message topic (subscribe): fedml_runid_serverID_clientID

        """
        logging.info("mqtt_s3.send_message: starting...{}".format(msg.to_string()))
        if self.client_id == 0:
            # server
            receiver_id = msg.get_receiver_id()

            # topic = "fedml" + "_" + "run_id" + "_0" + "_" + "client_id"
            topic = self._topic + str(0) + "_" + str(receiver_id)
            logging.info("mqtt_s3.send_message: msg topic = %s" % str(topic))

            payload = msg.get_params()
            model_params_obj = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
            message_key = topic + "_" + str(uuid.uuid4())
            if model_params_obj != "":
                # S3
                logging.info(
                    "mqtt_s3.send_message: S3+MQTT msg sent, s3 message key = %s"
                    % message_key
                )
                self.s3_storage.upload_model_file(message_key, model_params_obj)
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = message_key
                self._client.publish(topic, payload=json.dumps(payload))
            else:
                # pure MQTT
                logging.info("mqtt_s3.send_message: MQTT msg sent")
                self._client.publish(topic, payload=json.dumps(payload))

        else:
            raise Exception("This is only used for the server")

    def send_message_json(self, topic_name, json_message):
        self._client.publish(topic_name, payload=json_message)

    def handle_receive_message(self):
        self.run_loop_forever()
        # multiprocessing.Process(target=self.run_loop_forever).start()
        # self.is_running = True
        # while self.is_running:
        #     time.sleep(0.003)
        # logging.info("mqtt_s3.handle_receive_message: completed...")

    def stop_receive_message(self):
        logging.info("mqtt_s3.stop_receive_message: stopping...")
        self._client.loop_stop()
        self._client.disconnect()

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


if __name__ == "__main__":

    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s, %s)" % (msg_type, msg_params.to_string()))

    mqtt_config = (
        "../../../../fedml_experiments/distributed/fedavg_cross_silo/mqtt_config.yaml"
    )
    s3_config = (
        "../../../../fedml_experiments/distributed/fedavg_cross_silo/s3_config.yaml"
    )
    client = MqttS3MNNCommManager(
        mqtt_config, s3_config, topic="fedml_168_", client_id=1, client_num=1
    )
    client.add_observer(Obs())
    time.sleep(3)
    print("client ID:%s" % client.client_id)

    message = Message(0, 1, 2)
    message.add_params("key1", 1)
    client.send_message(message)

    time.sleep(10)
    print("client, finished to send...")
