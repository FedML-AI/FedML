import fedml

import requests

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.core.mlops.mlops_configs import MLOpsConfigs


class FedMLResourceManager(Singleton):
    def __init__(self):
        self.config_version = fedml.get_env_version()

    @staticmethod
    def get_instance():
        return FedMLResourceManager()

    def check_heartbeat(self, api_key):
        heartbeat_url = ServerConstants.get_heartbeat_url()
        heartbeat_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        heartbeat_json = {
            "apiKey": api_key
        }

        args = {"config_version": self.config_version}
        cert_path = MLOpsConfigs.get_cert_path_with_version()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    heartbeat_url, verify=True, headers=heartbeat_api_headers, json=heartbeat_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    heartbeat_url, verify=True, headers=heartbeat_api_headers, json=heartbeat_json
                )
        else:
            response = requests.post(heartbeat_url, headers=heartbeat_api_headers, json=heartbeat_json)
        if response.status_code != 200:
            print(f"Check heartbeat with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            return False
        else:
            resp_data = response.json()
            code = resp_data.get("code", "")
            message = resp_data.get("message", "")
            data = resp_data.get("data", False)
            if code == "SUCCESS" and data is True:
                return True

        return False

    def show_resource_type(self):
        resource_url = ServerConstants.get_resource_url()
        args = {"config_version": self.config_version}
        cert_path = MLOpsConfigs.get_cert_path_with_version()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.get(
                    resource_url, verify=True)
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.get(
                    resource_url, verify=True)
        else:
            # the server only allows GET
            response = requests.get(resource_url)
        if response.status_code != 200:
            print(f"Get resource type with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            pass
        else:
            resp_data = response.json()
            code = resp_data.get("code", "")
            message = resp_data.get("message", "")
            data = resp_data.get("data", None)
            if code == "SUCCESS" and data is not None:
                resource_list = list()
                for resource_item in data:
                    gpu_type = resource_item.get("gpuType", None)
                    resource_type = resource_item.get("resourceType", None)
                    resource_list.append((resource_type, gpu_type))
                return resource_list
            else:
                print(f"Get resource type with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")

        return None
