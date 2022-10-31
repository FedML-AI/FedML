import time

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

    @staticmethod
    def check_open_connection():
        args = {"config_version": "release"}
        try:
            mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
        except Exception as e:
            return False

        return True

    @staticmethod
    def check_s3_connection():
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
    def check_mqtt_connection():
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
                    return False;
                time.sleep(1)

            mqtt_mgr.disconnect()
            mqtt_mgr.loop_stop()
            return True
        except Exception as e:
            print("MQTT connect exception: {}".format(str(e)))
            return False

        return False

    def on_mqtt_connected(self, mqtt_client_object):
        self.is_mqtt_connected = True
        pass

    def on_mqtt_disconnected(self, mqtt_client_object):
        self.is_mqtt_connected = False


if __name__ == "__main__":
    pass

