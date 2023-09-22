import requests

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.core.mlops.mlops_configs import MLOpsConfigs

from fedml.computing.scheduler.comm_utils.security_utils import get_api_key, save_api_key


class FedMLClusterModelList(object):
    def __init__(self, cluster_list_json):
        self.cluster_list = list()
        for cluster_obj_json in cluster_list_json:
            cluster_obj = FedMLClusterModel(cluster_obj_json)
            self.cluster_list.append(cluster_obj)


class FedMLClusterModel(object):
    def __init__(self, cluster_json):
        self.cluster_id = cluster_json["id"]
        self.cluster_name = cluster_json["shortName"]
        self.status = cluster_json["status"]


class FedMLClusterManager(Singleton):

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLClusterManager()

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def kill_clusters(self, cluster_names=()):
        cluster_stop_url = ServerConstants.get_cluster_stop_url(self.config_version)
        cluster_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        cluster_list_json = {'cluster_names': list(set(cluster_names)), "user_api_key": get_api_key()}
        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    cluster_stop_url, verify=True, headers=cluster_api_headers, json=cluster_list_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    cluster_stop_url, verify=True, headers=cluster_api_headers, json=cluster_list_json
                )
        else:
            response = requests.post(
                cluster_stop_url, verify=True, headers=cluster_api_headers, json=cluster_list_json
            )
        if response.status_code != 200:
            print(f"Stop cluster with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            return False
        else:
            resp_data = response.json()
            code = resp_data.get("code", None)
            data = resp_data.get("data", None)
            if code is None or data is None or code == "FAILURE":
                print(f"Stop job with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return False

        return True

    def list_clusters(self, cluster_names=()):
        cluster_list_result = None
        cluster_list_url = ServerConstants.get_cluster_list_url(self.config_version)
        cluster_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        cluster_list_json = {'cluster_names': list(set(cluster_names)), "user_api_key": get_api_key()}
        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    cluster_list_url, verify=True, headers=cluster_api_headers, json=cluster_list_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    cluster_list_url, verify=True, headers=cluster_api_headers, json=cluster_list_json
                )
        else:
            response = requests.post(
                cluster_list_url, verify=True, headers=cluster_api_headers, json=cluster_list_json
            )
        if response.status_code != 200:
            print(
                f"Cluster list with response.status_code = {response.status_code}, response.content: {response.content}")
            pass
        else:
            resp_data = response.json()
            code = resp_data.get("code", None)
            data = resp_data.get("data", None)
            if code is None or data is None or code == "FAILURE":
                print(f"List job with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return None
            cluster_list_result = FedMLClusterModelList(data)
        return cluster_list_result
