import argparse
import json
import time
import uuid
from threading import Thread

from ...core.distributed.communication.mqtt_s3 import MqttS3MultiClientsCommManager
from ...core.mlops.mlops_configs import MLOpsConfigs
from ...core.distributed.communication.s3.remote_storage import S3Storage
from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance


class ClientDiagnosis(Singleton):
    def __init__(self):
        self.is_mqtt_connected = False
        self.mqtt_mgr = None
        self.test_mqtt_msg_process = None

    @staticmethod
    def check_open_connection(args=None):
        if args is None:
            args = {"config_version": "release"}
        try:
            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
        except Exception as e:
            return False

        return True

    @staticmethod
    def check_s3_connection(args=None):
        if args is None:
            args = {"config_version": "release"}
        try:
            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
            s3_storage = S3Storage(s3_config)
            download_ret = s3_storage.test_s3_base_cmds("d31df596c32943c64015a7e2d6e0d5a4", "test-base-cmds")
            if download_ret:
                return True
        except Exception as e:
            return False

        return False

    @staticmethod
    def check_mqtt_connection(args=None):
        if args is None:
            args = {"config_version": "release"}
        try:
            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
            mqtt_mgr = MqttManager(
                mqtt_config["BROKER_HOST"],
                mqtt_config["BROKER_PORT"],
                mqtt_config["MQTT_USER"],
                mqtt_config["MQTT_PWD"],
                mqtt_config["MQTT_KEEPALIVE"],
                "fedml-diagnosis-id"
            )
            diagnosis = ClientDiagnosis()
            diagnosis.is_mqtt_connected = False
            mqtt_mgr.add_connected_listener(diagnosis.on_mqtt_connected)
            mqtt_mgr.add_disconnected_listener(diagnosis.on_mqtt_disconnected)
            mqtt_mgr.connect()
            mqtt_mgr.loop_start()

            count = 0
            while not diagnosis.is_mqtt_connected:
                count += 1
                if count >= 15:
                    return False
                time.sleep(1)

            mqtt_mgr.disconnect()
            mqtt_mgr.loop_stop()
            return True
        except Exception as e:
            print("MQTT connect exception: {}".format(str(e)))
            return False

        return False

    @staticmethod
    def check_mqtt_s3_communication_backend(args=None):
        if args is None:
            parser = argparse.ArgumentParser(description="FedML")
            parser.add_argument("--config_version", type=str, default="release")
            parser.add_argument("--client_id_list", type=str, default="[1]")
            parser.add_argument("--dataset", type=str, default="default")
            parser.add_argument("--rank", type=int, default=1)
            args, unknown = parser.parse_known_args()
        try:
            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
            com_manager = MqttS3MultiClientsCommManager(
                mqtt_config,
                s3_config,
                topic="fedml-diagnosis-id-" + str(uuid.uuid4()),
                client_rank=1,
                client_num=1,
                args=args,
            )
            diagnosis = ClientDiagnosis()
            diagnosis.is_mqtt_connected = False
            com_manager.add_observer(diagnosis)

            # if self.test_mqtt_msg_process is None:
            #     self.test_mqtt_msg_process = Thread(target=self.send_test_mqtt_msg)
            #     self.test_mqtt_msg_process.start()

            # count = 0
            # while count < 15:
            #     com_manager.send_message_json("/fedml/" + com_manager.topic + "/check_mqtt_s3", '{"id":1}')
            #     count += 1
            #     time.sleep(1)

            com_manager.run_loop_forever()

            return True
        except Exception as e:
            print("mqtt_s3_communication_backend connect exception: {}".format(str(e)))
            return False

        return False

    @staticmethod
    def check_mqtt_connection_with_daemon_mode(args=None):
        if args is None:
            args = {"config_version": "release"}
        try:
            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
            mqtt_mgr = MqttManager(
                mqtt_config["BROKER_HOST"],
                mqtt_config["BROKER_PORT"],
                mqtt_config["MQTT_USER"],
                mqtt_config["MQTT_PWD"],
                mqtt_config["MQTT_KEEPALIVE"],
                "fedml-diagnosis-id-" + str(uuid.uuid4())
            )
            diagnosis = ClientDiagnosis()
            mqtt_mgr.add_connected_listener(diagnosis.on_test_mqtt_connected)
            mqtt_mgr.add_disconnected_listener(diagnosis.on_test_mqtt_disconnected)
            diagnosis.is_mqtt_connected = False
            diagnosis.mqtt_mgr = mqtt_mgr
            mqtt_mgr.connect()

            mqtt_mgr.loop_forever()
        except Exception as e:
            print("MQTT connect exception: {}".format(str(e)))
            return False

        return False

    def on_mqtt_connected(self, mqtt_client_object):
        self.is_mqtt_connected = True
        pass

    def on_mqtt_disconnected(self, mqtt_client_object):
        self.is_mqtt_connected = False

    def on_test_mqtt_connected(self, mqtt_client_object):
        self.is_mqtt_connected = True

        print("on_test_mqtt_connected")
        topic_test_mqtt_msg = "fedml/" + str(self.mqtt_mgr._client_id) + "/test_mqtt_msg"
        self.mqtt_mgr.add_message_listener(topic_test_mqtt_msg, self.callback_test_mqtt_msg)
        mqtt_client_object.subscribe(topic_test_mqtt_msg)

        if self.test_mqtt_msg_process is None:
            self.test_mqtt_msg_process = Thread(target=self.send_test_mqtt_msg)
            self.test_mqtt_msg_process.start()

    def on_test_mqtt_disconnected(self, mqtt_client_object):
        self.is_mqtt_connected = False

        print("on_test_mqtt_disconnected")

        topic_test_mqtt_msg = "fedml/" + str(self.mqtt_mgr._client_id) + "/test_mqtt_msg"
        self.mqtt_mgr.remove_message_listener(topic_test_mqtt_msg)
        mqtt_client_object.subscribe(topic_test_mqtt_msg)

    def callback_test_mqtt_msg(self, topic, payload):
        from time import strftime, localtime
        current_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
        print("[{}] Received test mqtt message, topic: {}, payload: {}.".format(current_time, topic, payload))

    def send_test_mqtt_msg(self):
        while True:
            topic_test_mqtt_msg = "fedml/" + str(self.mqtt_mgr._client_id) + "/test_mqtt_msg"
            test_mqtt_msg_payload = {"id": self.mqtt_mgr._client_id, "msg": topic_test_mqtt_msg}

            self.mqtt_mgr.send_message(topic_test_mqtt_msg, json.dumps(test_mqtt_msg_payload))
            time.sleep(60*4)

    def receive_message(self, msg_type, msg_params) -> None:
        print("fedml diagnosis received msg type {}, msg_params {}".format(msg_type, msg_params))


if __name__ == "__main__":
    pass

