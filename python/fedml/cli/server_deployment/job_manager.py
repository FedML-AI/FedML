import argparse
import json
import uuid

import requests

from ...core.common.singleton import Singleton
from .server_constants import ServerConstants
from ...core.mlops.mlops_configs import MLOpsConfigs


class FedMLJobManager(Singleton):

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLJobManager()

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def start_job(self, platform, project_name, application_name, devices, user_id, user_api_key):
        result = self.start_job_api(platform, project_name, application_name, devices, user_id, user_api_key)
        if result is None:
            return False

        return True

    def start_job_api(self, platform, project_name, application_name, devices, user_id, user_api_key):
        job_start_result = None
        jot_start_url = ServerConstants.get_job_start_url(self.config_version)
        job_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        job_start_json = {
            "platformType": platform,
            "applicationName": application_name,
            "devices": json.loads(devices),
            "name": "FedML-CLI-Job-" + str(uuid.uuid4()),
            "projectName": project_name,
            "urls": [],
            "userId": user_id,
            "apiKey": user_api_key
        }
        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    jot_start_url, verify=True, headers=job_api_headers, json=job_start_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    jot_start_url, verify=True, headers=job_api_headers, json=job_start_json
                )
        else:
            response = requests.post(jot_start_url, headers=job_api_headers, json=job_start_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            job_start_result = resp_data

        return job_start_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cf", "-c", help="config file")
    parser.add_argument("--role", "-r", type=str, default="client", help="role")
    in_args = parser.parse_args()
