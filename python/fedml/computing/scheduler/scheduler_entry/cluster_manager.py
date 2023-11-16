from typing import List

import requests
import json

import fedml
from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLGpuDevices
from fedml.core.mlops.mlops_configs import MLOpsConfigs

from fedml.computing.scheduler.comm_utils.security_utils import get_api_key


class ClusterConstants(object):
    JOB_ID = "jobId"
    API_KEY = "apiKey"
    CLUSTER_ID = "clusterId"
    MACHINE_SELECTED_LIST = "machineSelectedList"
    STATUS = "status"
    ID = "id"
    SHORT_NAME = "shortName"
    CLUSTER_NAME_LIST = "clusterNameList"
    AUTOSTOP_TIME = "time"


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
        self.config_version = fedml.get_env_version()

    @staticmethod
    def get_instance():
        return FedMLClusterManager()

    def start_clusters(self, cluster_names=()):
        cluster_start_url = ServerConstants.get_cluster_start_url()
        cluster_start_json = {'clusterNameList': list(set(cluster_names)), ClusterConstants.API_KEY: get_api_key()}
        response = self._request(cluster_start_url, cluster_start_json, self.config_version)
        data = self._get_data_from_response(command="Start", response=response)
        return True if data is not None else False

    def stop_clusters(self, cluster_names=()):
        cluster_stop_url = ServerConstants.get_cluster_stop_url()
        cluster_stop_json = {ClusterConstants.CLUSTER_NAME_LIST: list(set(cluster_names)),
                             ClusterConstants.API_KEY: get_api_key()}
        response = self._request(cluster_stop_url, cluster_stop_json, self.config_version)
        data = self._get_data_from_response(command="Stop", response=response)
        return True if data is not None else False

    def autostop_clusters(self, cluster_id: int, time: int) -> bool:
        cluster_autostop_url = ServerConstants.get_cluster_autostop_url()
        cluster_autostop_json = {ClusterConstants.CLUSTER_ID: cluster_id,
                                 ClusterConstants.AUTOSTOP_TIME: time}
        response = self._request(cluster_autostop_url, cluster_autostop_json, self.config_version)
        data = self._get_data_from_response(command="Autostop", response=response)
        return True if data is not None else False

    def kill_clusters(self, cluster_names=()):
        cluster_kill_url = ServerConstants.get_cluster_kill_url()
        cluster_list_json = {ClusterConstants.CLUSTER_NAME_LIST: list(set(cluster_names)),
                             ClusterConstants.API_KEY: get_api_key()}
        response = self._request(cluster_kill_url, cluster_list_json, self.config_version)
        data = self._get_data_from_response(command="Kill", response=response)
        return True if data is not None else False

    def list_clusters(self, cluster_names=()):
        cluster_list_url = ServerConstants.get_cluster_list_url()
        cluster_list_json = {ClusterConstants.CLUSTER_NAME_LIST: list(set(cluster_names)),
                             ClusterConstants.API_KEY: get_api_key()}
        response = self._request(cluster_list_url, cluster_list_json, self.config_version)
        data = self._get_data_from_response(command="List", response=response)
        return FedMLClusterModelList(data) if data is not None else data

    def confirm_and_start(self, run_id: str, cluster_id: str, gpu_matched: List[FedMLGpuDevices]):
        confirm_cluster_url = ServerConstants.get_cluster_confirm_url()
        confirm_cluster_json = self._get_cluster_confirm_json(run_id=run_id, cluster_id=cluster_id, gpu_matched=gpu_matched)
        response = self._request(url=confirm_cluster_url, json_data=confirm_cluster_json, config_version=self.config_version)
        data = self._get_data_from_response(command="Confirm", response=response)
        return True if data is not None else False

    @staticmethod
    def _request(url: str, json_data: dict, config_version: str):
        # print("json_data = ", json_data)
        args = {"config_version": config_version}
        cert_path = MLOpsConfigs.get_instance(args).get_cert_path_with_version()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, verify=True, headers=ServerConstants.API_HEADERS, json=json_data
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, verify=True, headers=ServerConstants.API_HEADERS, json=json_data
                )
        else:
            response = requests.post(
                url, verify=True, headers=ServerConstants.API_HEADERS, json=json_data
            )
        return response

    @staticmethod
    def _get_data_from_response(command: str, response):

        if response.status_code != 200:
            print(f"{command} cluster with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            return None
        else:
            resp_data = response.json()
            code = resp_data.get("code", None)
            data = resp_data.get("data", None)
            if code is None or data is None or code == "FAILURE":
                print(f"{command} cluster with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return None

        return data

    @staticmethod
    def _get_cluster_confirm_json(run_id: str, cluster_id: str, gpu_matched: List[FedMLGpuDevices]):
        selected_machines_list = list()
        for gpu_machine in gpu_matched:
            selected_machine_json = {
                "gotGpuCount": int(gpu_machine.gpu_count),
                ClusterConstants.ID: gpu_machine.gpu_id
            }
            # print(f"gotGpuCount = {gpu_machine.gpu_count}")
            selected_machines_list.append(selected_machine_json)

        confirm_cluster_dict = {ClusterConstants.JOB_ID: str(run_id),
                                ClusterConstants.CLUSTER_ID: str(cluster_id),
                                ClusterConstants.MACHINE_SELECTED_LIST: selected_machines_list,
                                ClusterConstants.API_KEY: get_api_key()}

        confirm_cluster_json_str = json.dumps(confirm_cluster_dict)
        confirm_cluster_json = json.loads(confirm_cluster_json_str)
        return confirm_cluster_json
