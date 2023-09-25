from typing import List

import requests

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLGpuDevices
from fedml.core.mlops.mlops_configs import MLOpsConfigs

from fedml.computing.scheduler.comm_utils.security_utils import get_api_key, save_api_key


class ClusterConstants(object):
    API_KEY = "apiKey"
    CLUSTER_ID = "clusterId"
    MACHINE_SELECTED_LIST = "machineSelectedList"
    STATUS = "status"
    ID = "id"
    SHORT_NAME = "shortName"


class FedMLClusterModelList(object):
    def __init__(self, cluster_list_json):
        self.cluster_list = list()
        for cluster_obj_json in cluster_list_json:
            cluster_obj = FedMLClusterModel(cluster_obj_json)
            self.cluster_list.append(cluster_obj)


class FedMLClusterModel(object):
    def __init__(self, cluster_json):
        self.cluster_id = cluster_json[ClusterConstants.ID]
        self.cluster_name = cluster_json[ClusterConstants.SHORT_NAME]
        self.status = cluster_json[ClusterConstants.STATUS]


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
        cluster_api_headers = ServerConstants.API_HEADERS
        cluster_list_json = {'clusterNameList': list(set(cluster_names)), ClusterConstants.API_KEY: get_api_key()}
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
                print(f"Stop cluster with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return False

        return True

    def list_clusters(self, cluster_names=()):
        cluster_list_result = None
        cluster_list_url = ServerConstants.get_cluster_list_url(self.config_version)
        cluster_api_headers = ServerConstants.API_HEADERS
        cluster_list_json = {'cluster_names': list(set(cluster_names)), ClusterConstants.API_KEY: get_api_key()}
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
                print(f"List cluster with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return None
            cluster_list_result = FedMLClusterModelList(data)
        return cluster_list_result

    def confirm_cluster(self, cluster_id: str, gpu_matched: List[FedMLGpuDevices]):
        confirm_cluster_url = ServerConstants.get_cluster_confirm_url(self.config_version)
        args = {"config_version": self.config_version}
        selected_machines_list = list()
        for gpu_machine in gpu_matched:
            selected_machine_json = {
                ClusterConstants.ID: gpu_machine.gpu_id,
                "got_gpt_count": gpu_machine.gpu_count
            }
            selected_machines_list.append(selected_machine_json)

        confirm_cluster_json = {ClusterConstants.CLUSTER_ID: cluster_id,
                                ClusterConstants.MACHINE_SELECTED_LIST: selected_machines_list,
                                ClusterConstants.API_KEY: get_api_key()}

        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)

        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    confirm_cluster_url, verify=True, headers=ServerConstants.API_HEADERS, json=confirm_cluster_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    confirm_cluster_url, verify=True, headers=ServerConstants.API_HEADERS, json=confirm_cluster_json
                )
        else:
            response = requests.post(
                confirm_cluster_url, verify=True, headers=ServerConstants.API_HEADERS, json=confirm_cluster_json
            )
        if response.status_code != 200:
            print(f"Cluster confirmation errored with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            return False
        else:
            resp_data = response.json()
            code = resp_data.get("code", None)
            data = resp_data.get("data", None)
            if code is None or data is None or code == "FAILURE":
                print(f"Cluster confirmation errored with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return False

        return True
