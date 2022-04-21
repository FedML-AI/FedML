# -*-coding:utf-8-*-
import json
import logging
import time
import traceback
import uuid
from typing import List

import paho.mqtt.client as mqtt
import torch
import yaml

from fedml_api.model.mobile.model_transfer import pytorch_mnn, mnn_pytorch
from fedml_core.distributed.communication.base_com_manager import BaseCommunicationManager
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.communication.observer import Observer
from .remote_storage import S3Storage


class MqttS3MultiClientsCommManager(BaseCommunicationManager):
    def __init__(
            self, config_path, s3_config_path, topic="fedml", client_id=0, client_num=0,
            args=None, bind_port=0
    ):
        client_objects_str = str(args.client_objects).replace('"', '\"')
        client_objects_str = client_objects_str.replace('\'', '')
        logging.info("origin client object " + str(args.client_objects))
        logging.info("client object " + client_objects_str)
        self.client_objects = json.loads(client_objects_str)

        self._topic = "fedml_" + topic + "_"
        self.s3_storage = S3Storage(s3_config_path)
        self.client_real_ids = []
        if args.client_ids is not None:
            logging.info("MqttS3CommManager args client_ids: " + str(args.client_ids))
            self.client_real_ids = json.loads(args.client_ids)
        self.origin_mnn_file = args.model_file_path
        if args.silo_rank == 0:
            self.edge_id = 0
        else:
            if len(self.client_real_ids) == 1:
                self.edge_id = self.client_real_ids[0]
            else:
                self.edge_id = 0

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
        # Construct a Client
        self.mqtt_connection_id = mqtt.base62(uuid.uuid4().int, padding=22)
        self._client = mqtt.Client(client_id=str(self.mqtt_connection_id), clean_session=True)
        self._client.enable_logger()

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        # self._client.on_log = self.on_log

        self._client.username_pw_set(self.mqtt_user, self.mqtt_pwd)
        _will_msg = {"ID": f"{self.mqtt_connection_id}", "stat": "Online"}
        self._client.will_set("W/topic", payload=json.dumps(_will_msg), qos=0, retain=True)

        logging.info("mqtt_s3.init: connecting to MQTT server(local port %d..." % bind_port)
        self._client.connect(self.broker_host, self.broker_port, 180, bind_port=bind_port)

    def get_client_os_type(self, client_id):
        for client_obj in self.client_objects:
            os_type = client_obj.get("os_type", Message.MSG_CLIENT_OS_LINUX)
            edge_id = client_obj.get("id", None)
            if str(client_id) != str(edge_id):
                continue
            return os_type

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
        logging.info("mqtt_s3.on_connect: connection returned with result code:" + str(rc))
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

            self._notify_connection_ready()
            logging.info("mqtt_s3.on_connect: server subscribes")
        else:
            # client
            real_topic = self._topic + str(0) + "_" + str(self.client_real_ids[0])
            result, mid = client.subscribe(real_topic, 0)
            self._unacked_sub.append(mid)

            self._notify_connection_ready()

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
        logging.info("mqtt_s3.on_disconnect: disconnection returned result %s, user data %s" % (str(rc), str(userdata)))

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        logging.info("mqtt_s3.onSubscribe: mid = %s" % str(mid))
        self._unacked_sub.remove(mid)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _notify_connection_ready(self):
        msg_params = Message()
        msg_type = 0
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

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
        sender_id = payload_obj.get(Message.MSG_ARG_KEY_SENDER, "")
        receiver_id = payload_obj.get(Message.MSG_ARG_KEY_RECEIVER, "")
        s3_key_str = payload_obj.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
        s3_key_str = str(s3_key_str).strip(" ")
        if self.client_id == 0:
            channel_id = sender_id
        else:
            channel_id = self.edge_id

        if s3_key_str != "":
            logging.info("mqtt_s3.on_message: use s3 pack, s3 message key %s" % s3_key_str)

            # read S3 object
            # s3_obj = self.s3_storage.read_json(s3_key_str)
            # model_params = str(s3_obj, encoding="utf-8")
            # model_params = json.loads(model_params)
            if self.get_client_os_type(channel_id) == Message.MSG_CLIENT_OS_ANDROID:
                logging.info("mqtt_s3.on_message: from android client.")
                mnn_file = self.s3_storage.read_model_with_file(s3_key_str)
                model_params = self.convert_mnn_file_to_pytorch_model(mnn_file)
            else:
                logging.info("mqtt_s3.on_message: from python client.")
                model_params = self.s3_storage.read_model(s3_key_str)

            logging.info("mqtt_s3.on_message: model params length %d" % len(model_params))

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
        logging.info("mqtt_s3.send_message: starting...")
        sender_id = msg.get_sender_id()
        receiver_id = msg.get_receiver_id()
        if self.client_id == 0:
            # topic = "fedml" + "_" + "run_id" + "_0" + "_" + "client_id"
            topic = self._topic + str(0) + "_" + str(receiver_id)
            logging.info("mqtt_s3.send_message: msg topic = %s" % str(topic))

            payload = msg.get_params()
            model_params_obj = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
            message_key = topic + "_" + str(uuid.uuid4())
            if model_params_obj != "":
                # S3
                logging.info("mqtt_s3.send_message: S3+MQTT msg sent, s3 message key = %s" % message_key)
                # self.s3_storage.write_json(message_key, json.dumps(model_params_obj))
                # model_uploaded = False
                # for model_params_key_item in self.model_params_key_map:
                #     if model_params_obj == model_params_key_item["obj"]:
                #         model_uploaded = True
                #         model_params_key_url = model_params_key_item
                #         break
                # if not model_uploaded:
                if self.get_client_os_type(receiver_id) == Message.MSG_CLIENT_OS_ANDROID:
                    logging.info("mqtt_s3.send_message: to android client.")
                    mnn_file = self.convert_pytorch_model_to_mnn_file(model_params_obj)
                    model_url = self.s3_storage.write_model_with_file(message_key, mnn_file)
                else:
                    logging.info("mqtt_s3.send_message: to python client.")
                    model_url = self.s3_storage.write_model(message_key, model_params_obj)
                model_params_key_url = {"key": message_key, "url": model_url, "obj": model_params_obj}
                #self.model_params_key_map.append(model_params_key_url)
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params_key_url["key"]
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL] = model_params_key_url["url"]
                self._client.publish(topic, payload=json.dumps(payload))
            else:
                # pure MQTT
                logging.info("mqtt_s3.send_message: MQTT msg sent")
                self._client.publish(topic, payload=json.dumps(payload))

        else:
            # client
            topic = self._topic + str(msg.get_sender_id())
            message_key = topic + "_" + str(uuid.uuid4())

            payload = msg.get_params()
            model_params_obj = payload.get(Message.MSG_ARG_KEY_MODEL_PARAMS, "")
            if model_params_obj != "":
                # S3
                logging.info("mqtt_s3.send_message: S3+MQTT msg sent, message_key = %s" % message_key)
                # self.s3_storage.write_json(message_key, json.dumps(model_params_obj))
                # model_uploaded = False
                # for model_params_key_item in self.model_params_key_map:
                #     if model_params_obj == model_params_key_item["obj"]:
                #         model_uploaded = True
                #         model_params_key_url = model_params_key_item
                #         break
                # if not model_uploaded:
                if self.get_client_os_type(sender_id) == Message.MSG_CLIENT_OS_ANDROID:
                    logging.info("mqtt_s3.send_message: to android client.")
                    mnn_file = self.convert_pytorch_model_to_mnn_file(model_params_obj)
                    model_url = self.s3_storage.write_model_with_file(message_key, mnn_file)
                else:
                    logging.info("mqtt_s3.send_message: to python client.")
                    model_url = self.s3_storage.write_model(message_key, model_params_obj)
                model_params_key_url = {"key": message_key, "url": model_url, "obj": model_params_obj}
                #self.model_params_key_map.append(model_params_key_url)
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params_key_url["key"]
                payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL] = model_params_key_url["url"]
                self._client.publish(topic, payload=json.dumps(payload))
            else:
                logging.info("mqtt_s3.send_message: MQTT msg sent")
                self._client.publish(topic, payload=json.dumps(payload))

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

    def convert_pytorch_model_to_mnn_file(self, pytorch_model_params):
        pytorch_model_file = "/tmp/" + str(uuid.uuid4()) + ".ckpt"
        torch.save(pytorch_model_params, pytorch_model_file)
        logging.info("convert_pytorch_model_to_mnn_file: torch file " + pytorch_model_file)

        mnn_model_file = "/tmp/" + str(uuid.uuid4()) + ".mnn"
        pytorch_mnn(pytorch_model_file, self.origin_mnn_file, mnn_model_file)
        self.origin_mnn_file = mnn_model_file
        logging.info("convert_pytorch_model_to_mnn_file: mnn file " + mnn_model_file)

        return mnn_model_file

    def convert_mnn_file_to_pytorch_model(self, mnn_model_file):
        pytorch_model_file = "/tmp/" + str(uuid.uuid4()) + ".ckpt"
        mnn_pytorch(mnn_model_file, pytorch_model_file)
        logging.info("convert_mnn_file_to_pytorch_model: torch file " + pytorch_model_file)

        return torch.load(pytorch_model_file)


if __name__ == "__main__":

    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s, %s)" % (msg_type, msg_params.to_string()))

    mqtt_config = "../../../../fedml_experiments/distributed/fedavg_cross_silo/mqtt_config.yaml"
    s3_config = "../../../../fedml_experiments/distributed/fedavg_cross_silo/s3_config.yaml"
    client = MqttS3MultiClientsCommManager(mqtt_config, s3_config, topic="fedml_168_", client_id=1, client_num=1)
    client.add_observer(Obs())
    time.sleep(3)
    print("client ID:%s" % client.client_id)

    message = Message(0, 1, 2)
    message.add_params("key1", 1)
    client.send_message(message)

    time.sleep(10)
    print("client, finished to send...")
