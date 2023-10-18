
import os
import time
import certifi
import requests
import fedml
from fedml.core.mlops.mlops_utils import MLOpsUtils


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance


class MLOpsConfigs(object):
    _config_instance = None

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            orig = super(MLOpsConfigs, cls)
            cls._instance = orig.__new__(cls)
            cls._instance.init()
        return cls._instance

    def __init__(self):
        pass

    def init(self):
        self.args = None

    @staticmethod
    def get_instance(args):
        if MLOpsConfigs._config_instance is None:
            MLOpsConfigs._config_instance = MLOpsConfigs()
            MLOpsConfigs._config_instance.args = args

        return MLOpsConfigs._config_instance

    def get_request_params(self):
        url = fedml._get_backend_service()
        url = f"{url}/fedmlOpsServer/configs/fetch"
        cert_path = None
        if str(url).startswith("https://"):
            cur_source_dir = os.path.dirname(__file__)
            cert_path = os.path.join(
                cur_source_dir, "ssl", "open-" + fedml.get_env_version() + ".fedml.ai_bundle.crt"
            )

        return url, cert_path

    def get_cert_path_with_version(self):
        url = fedml._get_backend_service()
        version = fedml.get_env_version()
        cert_path = None
        if str(url).startswith("https://"):
            cur_source_dir = os.path.dirname(__file__)
            cert_path = os.path.join(
                cur_source_dir, "ssl", "open-" + version + ".fedml.ai_bundle.crt"
            )
        return cert_path

    @staticmethod
    def get_root_ca_path():
        cur_source_dir = os.path.dirname(__file__)
        cert_path = os.path.join(
            cur_source_dir, "ssl", "open-root-ca.crt"
        )
        return cert_path

    @staticmethod
    def install_root_ca_file():
        ca_file = certifi.where()
        open_root_ca_path = MLOpsConfigs.get_root_ca_path()
        with open(open_root_ca_path, 'rb') as infile:
            open_root_ca_file = infile.read()
        with open(ca_file, 'ab') as outfile:
            outfile.write(open_root_ca_file)

    def fetch_configs(self):
        url, cert_path = self.get_request_params()
        json_params = {"config_name": ["mqtt_config", "s3_config", "ml_ops_config"],
                       "device_send_time": int(time.time() * 1000)}

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
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
                MLOpsConfigs.install_root_ca_file()
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
                MLOpsConfigs.install_root_ca_file()
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
        # print("url = {}, cert_path = {}".format(url, cert_path))
        json_params = {
            "config_name": ["mqtt_config", "s3_config", "ml_ops_config", "docker_config"],
            "device_send_time": int(time.time() * 1000)
        }

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
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

    @staticmethod
    def fetch_all_configs_with_version():
        url = fedml._get_backend_service()
        url = f"{url}/fedmlOpsServer/configs/fetch"
        cert_path = None
        if str(url).startswith("https://"):
            cur_source_dir = os.path.dirname(__file__)
            cert_path = os.path.join(
                cur_source_dir, "ssl", "open-" + fedml.get_env_version() + ".fedml.ai_bundle.crt"
            )

        json_params = {
            "config_name": ["mqtt_config", "s3_config", "ml_ops_config", "docker_config"],
            "device_send_time": int(time.time() * 1000)
        }

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
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
