import os
import time
from enum import Enum



import certifi
import requests
import fedml
from fedml.core.mlops.mlops_utils import MLOpsUtils


class Configs(Enum):
    MQTT_CONFIG = "mqtt_config"
    S3_CONFIG = "s3_config"
    ML_OPS_CONFIG = "ml_ops_config"
    DOCKER_CONFIG = "docker_config"
    WEB3_CONFIG = "web3_config"
    THETASTORE_CONFIG = "thetastore_config"
    R2_CONFIG = "r2_config"


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance


class MLOpsConfigs(object):
    _instance = None

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            orig = super(MLOpsConfigs, cls)
            cls._instance = orig.__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        pass

    @staticmethod
    def get_request_params():
        url = fedml._get_backend_service()
        url = f"{url}/fedmlOpsServer/configs/fetch"
        cert_path = None
        if str(url).startswith("https://"):
            cur_source_dir = os.path.dirname(__file__)
            cert_path = os.path.join(
                cur_source_dir, "ssl", "open-" + fedml.get_env_version() + ".fedml.ai_bundle.crt"
            )

        return url, cert_path

    @staticmethod
    def _request(request_url: str, request_json: dict, request_headers=None, cert_path=None) -> requests.Response:
        if request_headers is None:
            request_headers = {}
        request_headers["Connection"] = "close"
        request_headers["Content-Type"] = "application/json"

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    request_url, json=request_json, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    request_url, json=request_json, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(
                request_url, json=request_json, headers={"content-type": "application/json", "Connection": "close"}
            )
        return response

    @staticmethod
    def get_cert_path_with_version():
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

    @staticmethod
    def _fetch_configs(configs) -> dict:
        url, cert_path = MLOpsConfigs.get_request_params()
        request_configs = {Configs.ML_OPS_CONFIG}
        request_configs = request_configs.union(configs)
        json_params = {"config_name": [config.value for config in request_configs],
                       "device_send_time": int(time.time() * 1000)}
        response = MLOpsConfigs._request(request_url=url, request_json=json_params, cert_path=cert_path)
        status_code = response.json().get("code")
        result = {}
        if status_code == "SUCCESS":
            data = response.json().get("data")
            for config in configs:
                result[config] = data.get(config.value)
            mlops_config = data.get(Configs.ML_OPS_CONFIG.value)
            MLOpsUtils.calc_ntp_from_config(mlops_config)
        else:
            raise Exception("failed to fetch device configurations!")
        return result

    @staticmethod
    def fetch_web3_configs():
        fetched_configs = MLOpsConfigs._fetch_configs({Configs.MQTT_CONFIG, Configs.WEB3_CONFIG})
        return fetched_configs[Configs.MQTT_CONFIG], fetched_configs[Configs.WEB3_CONFIG]

    @staticmethod
    def fetch_thetastore_configs():
        fetched_configs = MLOpsConfigs._fetch_configs({Configs.MQTT_CONFIG, Configs.THETASTORE_CONFIG})
        return fetched_configs[Configs.MQTT_CONFIG], fetched_configs[Configs.THETASTORE_CONFIG]

    @staticmethod
    def fetch_remote_storage_configs():
        fetched_configs = MLOpsConfigs._fetch_configs({Configs.S3_CONFIG, Configs.R2_CONFIG})
        return fetched_configs

    @staticmethod
    def fetch_all_configs():
        fetched_configs = MLOpsConfigs._fetch_configs({Configs.MQTT_CONFIG, Configs.S3_CONFIG, Configs.ML_OPS_CONFIG,
                                                       Configs.DOCKER_CONFIG})
        return (fetched_configs[Configs.MQTT_CONFIG],
                fetched_configs[Configs.S3_CONFIG],
                fetched_configs[Configs.ML_OPS_CONFIG],
                fetched_configs[Configs.DOCKER_CONFIG])


if __name__ == "__main__":
    fedml.set_env_version("release")
    mqtt_config, s3_config, mlops_config, docker_config = MLOpsConfigs.fetch_all_configs()
    print(mqtt_config)
    print(s3_config)
    print(mlops_config)
    print(docker_config)
