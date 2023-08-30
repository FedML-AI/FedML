import argparse
import json
import time
import traceback
import uuid
from threading import Thread

from fedml.core.distributed.communication.message import Message
from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager
from fedml.core.distributed.communication.mqtt_s3 import MqttS3MultiClientsCommManager
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.core.mlops.mlops_configs import MLOpsConfigs


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
        self.test_mqtt_s3_backend_server_process = None
        self.test_mqtt_s3_backend_client_process = None
        self.test_mqtt_s3_com_manager_client = None
        self.test_mqtt_s3_com_manager_server = None

    @staticmethod
    def check_open_connection(args=None):
        if args is None:
            args = {"config_version": "release"}
        try:
            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
        except Exception as e:
            traceback.print_exc(e)
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
            traceback.print_exc(e)
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
                "FedML_Diagnosis_Normal_" + str(uuid.uuid4())
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
                if count > 15:
                    return False
                time.sleep(1)

            mqtt_mgr.disconnect()
            mqtt_mgr.loop_stop()
            return True
        except Exception as e:
            print("MQTT connect exception: {}".format(str(e)))
            traceback.print_exc(e)
            return False

    @staticmethod
    def check_mqtt_s3_communication_backend_server(run_id, args=None):
        if args is None:
            parser = argparse.ArgumentParser(description="FedML")
            parser.add_argument("--config_version", type=str, default="release")
            parser.add_argument("--client_id_list", type=str, default="[1]")
            parser.add_argument("--dataset", type=str, default="default")
            parser.add_argument("--rank", type=int, default=0)
            args, unknown = parser.parse_known_args()
            setattr(args, "run_id", run_id)
        try:
            diagnosis = ClientDiagnosis()

            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
            comm_server = MqttS3MultiClientsCommManager(
                mqtt_config,
                s3_config,
                topic="FedML_Diagnosis_CS_" + str(run_id),
                client_rank=0,
                client_num=1,
                args=args,
            )
            diagnosis.test_mqtt_s3_com_manager_server = comm_server
            comm_server.add_observer(diagnosis)

            if diagnosis.test_mqtt_s3_backend_server_process is None:
                diagnosis.test_mqtt_s3_backend_server_process = Thread(
                    target=diagnosis.send_test_mqtt_s3_backend_server_msg)
                diagnosis.test_mqtt_s3_backend_server_process.start()
            comm_server.mqtt_mgr.loop_forever()
            return True
        except Exception as e:
            print("mqtt_s3_communication_backend_server connect exception: {}".format(str(e)))
            traceback.print_exc(e)
            return False

    @staticmethod
    def check_mqtt_s3_communication_backend_client(run_id, args=None):
        if args is None:
            parser = argparse.ArgumentParser(description="FedML")
            parser.add_argument("--config_version", type=str, default="release")
            parser.add_argument("--client_id_list", type=str, default="[1]")
            parser.add_argument("--dataset", type=str, default="default")
            parser.add_argument("--rank", type=int, default=1)
            args, unknown = parser.parse_known_args()
            setattr(args, "run_id", run_id)
        try:
            diagnosis = ClientDiagnosis()

            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
            comm_client = MqttS3MultiClientsCommManager(
                mqtt_config,
                s3_config,
                topic="FedML_Diagnosis_CS_" + str(run_id),
                client_rank=1,
                client_num=1,
                args=args,
            )
            comm_client.add_observer(diagnosis)
            diagnosis.test_mqtt_s3_com_manager_client = comm_client

            if diagnosis.test_mqtt_s3_backend_client_process is None:
                diagnosis.test_mqtt_s3_backend_client_process = Thread(
                    target=diagnosis.send_test_mqtt_s3_backend_client_msg)
                diagnosis.test_mqtt_s3_backend_client_process.start()

            comm_client.mqtt_mgr.loop_forever()

            return True
        except Exception as e:
            print("mqtt_s3_communication_backend_client connect exception: {}".format(str(e)))
            traceback.print_exc(e)
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
                10,  # mqtt_config["MQTT_KEEPALIVE"],
                "FedML_Diagnosis_Daemon_" + str(uuid.uuid4())
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
            traceback.print_exc(e)
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

            ret = self.mqtt_mgr.send_message(topic_test_mqtt_msg, json.dumps(test_mqtt_msg_payload))
            print("send ret {}".format(str(ret)))
            time.sleep(3)

    def send_test_mqtt_s3_backend_server_msg(self):
        while True:
            time.sleep(2)
            message = Message(1, 0, 1)
            ret = self.test_mqtt_s3_com_manager_server.send_message(message)
            print("server is sending messages to client...")

    def send_test_mqtt_s3_backend_client_msg(self):
        while True:
            time.sleep(2)
            message = Message(2, 1, 0)
            ret = self.test_mqtt_s3_com_manager_client.send_message(message)
            print("client is sending messages to server...")

    def receive_message(self, msg_type, msg_params) -> None:
        print("fedml diagnosis received msg type {}, msg_params {}".format(msg_type, msg_params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", type=str, help="Diagnosis as client or server")
    parser.add_argument("--run_id", "-r", type=str, help="run id for client and server")
    args = parser.parse_args()
    diagnosis = ClientDiagnosis()
    if args.type == 'client':
        diagnosis.check_mqtt_s3_communication_backend_client(args.run_id)
    else:
        diagnosis.check_mqtt_s3_communication_backend_server(args.run_id)
