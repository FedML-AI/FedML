
import os
import time

import certifi
import requests
from fedml.core.mlops.mlops_utils import MLOpsUtils


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance


class ModelOpsConfigs(Singleton):
    _config_instance = None

    def __init__(self):
        self.args = None

    @staticmethod
    def get_instance(args):
        if ModelOpsConfigs._config_instance is None:
            ModelOpsConfigs._config_instance = ModelOpsConfigs()
            ModelOpsConfigs._config_instance.args = args

        return ModelOpsConfigs._config_instance

    def get_request_params(self, in_config_version="release"):
        url = "https://open.fedml.ai/fedmlOpsServer/configs/fetch"
        config_version = "release"
        if in_config_version is not None and in_config_version != "":
            # Setup config url based on selected version.
            config_version = in_config_version
            if config_version == "release":
                url = "https://open.fedml.ai/fedmlOpsServer/configs/fetch"
            elif config_version == "test":
                url = "https://open-test.fedml.ai/fedmlOpsServer/configs/fetch"
            elif config_version == "dev":
                url = "https://open-dev.fedml.ai/fedmlOpsServer/configs/fetch"
            elif config_version == "local":
                if hasattr(self.args, "local_server") and self.args.local_server is not None:
                    url = "http://{}:9000/fedmlOpsServer/configs/fetch".format(self.args.local_server)
                else:
                    url = "http://localhost:9000/fedmlOpsServer/configs/fetch"

        cert_path = None
        if str(url).startswith("https://"):
            cur_source_dir = os.path.dirname(__file__)
            cert_path = os.path.join(
                cur_source_dir, "ssl", "model-" + config_version + ".fedml.ai_bundle.crt"
            )

        return url, cert_path

    @staticmethod
    def get_root_ca_path():
        cur_source_dir = os.path.dirname(__file__)
        cert_path = os.path.join(cur_source_dir, "..", "..", "core", "mlops", "ssl", "open-root-ca.crt")
        return cert_path

    @staticmethod
    def install_root_ca_file():
        ca_file = certifi.where()
        open_root_ca_path = ModelOpsConfigs.get_root_ca_path()
        with open(open_root_ca_path, 'rb') as infile:
            open_root_ca_file = infile.read()
        with open(ca_file, 'ab') as outfile:
            outfile.write(open_root_ca_file)

    def fetch_configs(self, config_version="release"):
        url, cert_path = self.get_request_params(config_version)
        json_params = {"config_name": ["mqtt_config", "s3_config", "ml_ops_config"],
                       "device_send_time": int(time.time() * 1000)}

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(
                url, json=json_params, headers={"content-type": "application/json", "Connection": "close"}
            )

        status_code = response.json().get("code")
        if status_code == "SUCCESS":
            mqtt_config = response.json().get("data").get("mqtt_config")
            s3_config = response.json().get("data").get("s3_config")
            mlops_config = response.json().get("data").get("ml_ops_config")
            MLOpsUtils.calc_ntp_from_config(mlops_config)
        else:
            raise Exception("failed to fetch device configurations!")
        return mqtt_config, s3_config

    def fetch_web3_configs(self):
        url, cert_path = self.get_request_params()
        json_params = {"config_name": ["mqtt_config", "web3_config", "ml_ops_config"],
                       "device_send_time": int(time.time() * 1000)}

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(
                url, json=json_params, headers={"content-type": "application/json", "Connection": "close"}
            )

        status_code = response.json().get("code")
        if status_code == "SUCCESS":
            mqtt_config = response.json().get("data").get("mqtt_config")
            web3_config = response.json().get("data").get("web3_config")
            mlops_config = response.json().get("data").get("ml_ops_config")
            MLOpsUtils.calc_ntp_from_config(mlops_config)
        else:
            raise Exception("failed to fetch device configurations!")
        return mqtt_config, web3_config

    def fetch_thetastore_configs(self):
        url, cert_path = self.get_request_params()
        json_params = {"config_name": ["mqtt_config", "thetastore_config", "ml_ops_config"],
                       "device_send_time": int(time.time() * 1000)}

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(
                url, json=json_params, headers={"content-type": "application/json", "Connection": "close"}
            )

        status_code = response.json().get("code")
        if status_code == "SUCCESS":
            mqtt_config = response.json().get("data").get("mqtt_config")
            thetastore_config = response.json().get("data").get("thetastore_config")
            mlops_config = response.json().get("data").get("ml_ops_config")
            MLOpsUtils.calc_ntp_from_config(mlops_config)
        else:
            raise Exception("failed to fetch device configurations!")
        return mqtt_config, thetastore_config

    def fetch_all_configs(self):
        url, cert_path = self.get_request_params()
        json_params = {"config_name": ["mqtt_config", "s3_config", "ml_ops_config", "docker_config"],
                       "device_send_time": int(time.time() * 1000)}

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(
                url, json=json_params, headers={"content-type": "application/json", "Connection": "close"}
            )

        status_code = response.json().get("code")
        if status_code == "SUCCESS":
            mqtt_config = response.json().get("data").get("mqtt_config")
            s3_config = response.json().get("data").get("s3_config")
            mlops_config = response.json().get("data").get("ml_ops_config")
            docker_config = response.json().get("data").get("docker_config")
            MLOpsUtils.calc_ntp_from_config(mlops_config)
        else:
            raise Exception("failed to fetch device configurations!")

        return mqtt_config, s3_config, mlops_config, docker_config


if __name__ == "__main__":
    pass
