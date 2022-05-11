import os
import requests


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance


class MLOpsConfigs(Singleton):
    _config_instance = None

    def __init__(self):
        pass

    @staticmethod
    def get_instance(args):
        if MLOpsConfigs._config_instance is None:
            MLOpsConfigs._config_instance = MLOpsConfigs()
            MLOpsConfigs._config_instance.args = args

        return MLOpsConfigs._config_instance

    def fetch_configs(self):
        url = "https://open.fedml.ai/fedmlOpsServer/configs/fetch"
        if hasattr(self.args, "config_version") and self.args.config_version is not None:
            # Setup config url based on selected version.
            if self.args.config_version == "release":
                url = "https://open.fedml.ai/fedmlOpsServer/configs/fetch"
            elif self.args.config_version == "test":
                url = "http://open-test.fedml.ai/fedmlOpsServer/configs/fetch"
            elif self.args.config_version == "dev":
                url = "http://open-dev.fedml.ai/fedmlOpsServer/configs/fetch"
            elif self.args.config_version == "local":
                url = "http://localhost:9000/fedmlOpsServer/configs/fetch"

        json_params = {"config_name": ["mqtt_config", "s3_config"]}
        if str(url).startswith("https://"):
            cur_source_dir = os.path.dirname(__file__)
            cert_path = os.path.join(cur_source_dir, "ssl", "open.fedml.ai_bundle.crt")
            requests.session().verify = cert_path
            response = requests.post(url, json=json_params, verify=True, headers={'Connection': 'close'})
        else:
            response = requests.post(url, json=json_params, headers={'Connection': 'close'})
        status_code = response.json().get("code")
        if status_code == "SUCCESS":
            mqtt_config = response.json().get("data").get("mqtt_config")
            s3_config = response.json().get("data").get("s3_config")
        else:
            raise Exception("failed to fetch device configurations!")
        return mqtt_config, s3_config


if __name__ == "__main__":
    pass
