import os
import time
import uuid
import fedml

import requests
from requests import Response
from typing import List

from fedml.computing.scheduler.scheduler_entry.constants import Constants

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.core.mlops.mlops_configs import MLOpsConfigs
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLJobConfig
from enum import Enum


class FeatureEntryPoint(Enum):
    FEATURE_ENTRYPOINT_JOB_STORE_TRAIN_SUBTYPE = 1
    FEATURE_ENTRYPOINT_JOB_STORE_DEPLOY_SUBTYPE = 2
    FEATURE_ENTRYPOINT_JOB_STORE_FEDERATE_SUBTYPE = 3
    FEATURE_ENTRYPOINT_JOB_STORE_VECTOR_DB_SUBTYPE = 4

    FEATURE_ENTRYPOINT_STUDIO_LLM_FINE_TUNING = 5
    FEATURE_ENTRYPOINT_STUDIO_LLM_DEPLOYMENT = 6

    FEATURE_ENTRYPOINT_RUN_TRAIN = 7
    FEATURE_ENTRYPOINT_RUN_DEPLOY = 8
    FEATURE_ENTRYPOINT_RUN_FEDERATE = 9

    FEATURE_ENTRYPOINT_CLI = 10
    FEATURE_ENTRYPOINT_API = 11


class FedMLRunStartedModel(object):
    def __init__(self, response: Response, data: dict, project_name: str = None, application_name: str = None,
                 job_type: str = None,
                 inner_id: str = None, app_job_id: str = None, app_job_name: str = None):
        if data is not None:
            self.run_id = data.get("job_id", "0")
            self.run_name = data.get("job_name", None)
            self.project_id = data.get("project_id", None)
            self.status = data.get("status", None)
            self.status = data.get("code",
                                   Constants.MLOPS_CLIENT_STATUS_NOT_STARTED) if self.status is None else self.status
            self.run_url = data.get("job_url", data)
            self.gpu_matched = list()
            self.message = data.get("message", None)
            self.cluster_id = data.get("cluster_id", None)
            gpu_list_json = data.get("gpu_matched", None)
            if gpu_list_json is not None:
                for gpu_dev_json in gpu_list_json:
                    self.gpu_matched.append(FedMLGpuDevices(gpu_dev_json))
            self.started_time = data.get("started_time", time.time())
            self.user_check = data.get("user_check", True)
        else:
            self.run_id = "0"
            self.run_name = None
            self.project_id = ""
            self.run_url = data
            self.started_time = time.time()
            self.user_check = True
            self.status = None
            self.message = None
        self.status = response.status_code if self.status is None else self.status
        self.message = response.content if self.message is None else self.message
        self.inner_id = inner_id
        self.project_name = project_name
        self.application_name = application_name
        self.job_type = job_type
        self.app_job_id = app_job_id
        self.app_job_name = app_job_name


class FedMLRunConfig(object):

    def __init__(self, result: FedMLRunStartedModel = None, device_server: str = None,
                 device_edges: List[str] = None):
        self.run_id = result.run_id if result is not None else None
        self.run_name = result.run_name if result is not None else None
        self.project_name = result.project_name if result is not None else None
        self.application_name = result.application_name if result is not None else None
        self.job_type = result.job_type if result is not None else None
        self.project_id = result.project_id if result is not None else None
        self.app_job_id = result.app_job_id if result is not None else None
        self.app_job_name = result.app_job_name if result is not None else None
        self.device_server = device_server
        self.device_edges = device_edges


class FedMLGpuDevices(object):
    def __init__(self, gpu_device_json: dict):
        self.gpu_id = gpu_device_json.get("id", None)
        self.gpu_vendor = gpu_device_json.get("gpu_vendor", None)
        self.gpu_num = gpu_device_json.get("total_gpu_count", None)
        self.gpu_type = gpu_device_json.get("gpu_type", None)
        self.cost = gpu_device_json.get("cost", None)
        self.mem_size = gpu_device_json.get("gpu_mem", None)
        self.gpu_region = gpu_device_json.get("gpu_region", "DEFAULT")
        self.gpu_region = "DEFAULT" if self.gpu_region is None or self.gpu_region == "" else self.gpu_region
        self.cpu_count = gpu_device_json.get("cpu_count", None)
        self.cpu_count = None if self.cpu_count is not None and int(self.cpu_count) <= 0 else self.cpu_count
        self.gpu_count = gpu_device_json.get("got_gpu_count", -1)
        self.gpu_name = gpu_device_json.get("gpu_name", None)
        self.gpu_instance = self.gpu_name
        self.gpu_provider = gpu_device_json.get("gpu_provider", None)


class FedMLRunModelList(object):
    def __init__(self, run_list_json: dict):
        run_list_data = run_list_json.get("jobList", [])
        self.run_list = list()
        for run_obj_json in run_list_data:
            run_obj = FedMLRunModel(run_obj_json)
            self.run_list.append(run_obj)


class FedMLRunModel(object):
    def __init__(self, run_json: dict):
        self.run_id = run_json["id"]
        self.run_name = run_json["name"]
        self.status = run_json["status"]
        self.started_time = Constants.format_time_trimmed_tz(run_json.get("createTime", "0"))
        self.ended_time = Constants.format_time_trimmed_tz(run_json.get("endTime", "0"))
        self.running_time = run_json.get("spendTime", 0)
        self.compute_start_time = self.started_time
        self.compute_end_time = self.ended_time
        self.compute_duration = run_json.get("spendTime", 0)
        if self.compute_duration is not None and self.compute_duration != 'None':
            self.compute_duration = self.compute_duration / Constants.TIME_PER_HOUR_TO_MS
            self.compute_duration = round(self.compute_duration, 4)
        self.cost = run_json.get("cost", 0.0)
        self.run_url = run_json.get("jobUrl", "")
        gpu_machines = run_json.get("gpuMachines", None)
        self.device_infos = list()
        if gpu_machines is not None:
            for gpu_dev in gpu_machines:
                device_id = gpu_dev["deviceId"]
                gpu_count = gpu_dev.get("gpuCount", 0)
                brand_value = gpu_dev.get("brand", Constants.GPU_BRAND_MAPPING_INDEX_NVIDIA)
                brand = Constants.GPU_BRAND_MAPPING.get(brand_value,
                                                        Constants.GPU_BRAND_MAPPING[
                                                            Constants.GPU_BRAND_MAPPING_INDEX_OTHER])
                os_type = gpu_dev.get("osType", "")
                os_version = gpu_dev.get("osVersion", "")
                self.device_infos.append(f"Device Name: {device_id}, OS Type: {os_type}, OS Version: {os_version}, "
                                         f"Brand: {brand}, gpu count: {gpu_count}")


class FedMLRunLogModelList(object):
    def __init__(self, run_log_list_json):
        self.log_full_url = run_log_list_json.get("log_full_url", None)
        self.log_full_url = None if self.log_full_url is not None and self.log_full_url == "" else self.log_full_url
        log_devices_json = run_log_list_json.get("devices", [])
        self.log_devices = list()
        for log_dev in log_devices_json:
            self.log_devices.append(FedMLRunLogDeviceModel(log_dev))
        self.total_num = run_log_list_json.get("total_num", 0)
        self.total_pages = run_log_list_json.get("total_pages", 0)
        self.current_page = run_log_list_json.get("current_page", 0)
        self.log_lines = run_log_list_json.get("logs", [])


class FedMLRunLogDeviceModel(object):
    def __init__(self, run_log_device_json):
        self.log_url = run_log_device_json.get("log_url", None)
        self.log_url = None if self.log_url is not None and self.log_url == "" else self.log_url
        self.device_name = run_log_device_json.get("name", None)
        self.device_id = run_log_device_json.get("id", None)


class FedMLRunManager(Singleton):

    def __init__(self):
        self.config_version = fedml.get_env_version()

    @staticmethod
    def get_instance():
        return FedMLRunManager()

    def create_run(self, platform: str, job_config: FedMLJobConfig, device_server: str, device_edges: List[str],
                   api_key: str, cluster: str = None,
                   feature_entry_point: FeatureEntryPoint = None) -> FedMLRunStartedModel:
        run_create_json = self._get_run_create_json(platform=platform, project_name=job_config.project_name,
                                                    application_name=job_config.application_name,
                                                    device_server=device_server, device_edges=device_edges,
                                                    api_key=api_key, task_type=job_config.task_type,
                                                    app_job_id=job_config.job_id, app_job_name=job_config.job_name,
                                                    cluster=cluster, config_id=job_config.config_id,
                                                    feature_entry_point=feature_entry_point)
        response = self._request(request_url=ServerConstants.get_run_start_url(),
                                 request_json=run_create_json,
                                 config_version=self.config_version)
        response_data = self._get_data_from_response(response=response)
        inner_id = job_config.serving_endpoint_id \
            if job_config.task_type == Constants.JOB_TASK_TYPE_DEPLOY or \
               job_config.task_type == Constants.JOB_TASK_TYPE_SERVE else None
        run_start_result = FedMLRunStartedModel(response=response, data=response_data,
                                                project_name=job_config.project_name,
                                                application_name=job_config.application_name, inner_id=inner_id,
                                                app_job_id=job_config.job_id, app_job_name=job_config.job_name)
        return run_start_result

    def start_run(self, platform: str, create_run_result: FedMLRunStartedModel, device_server: str,
                  device_edges: List[str], api_key: str,
                  feature_entry_point: FeatureEntryPoint = None) -> FedMLRunStartedModel:
        run_start_result = None
        run_start_json = self._get_run_start_json(run_id=create_run_result.run_id, platform=platform,
                                                  project_name=create_run_result.project_name,
                                                  application_name=create_run_result.application_name,
                                                  device_server=device_server, device_edges=device_edges,
                                                  api_key=api_key, task_type=create_run_result.job_type,
                                                  app_job_id=create_run_result.app_job_id,
                                                  app_job_name=create_run_result.app_job_name,
                                                  feature_entry_point=feature_entry_point)

        response = self._request(request_url=ServerConstants.get_run_start_url(),
                                 request_json=run_start_json,
                                 config_version=self.config_version)

        response_data = self._get_data_from_response(response=response)

        run_start_result = FedMLRunStartedModel(response=response, data=response_data,
                                                project_name=create_run_result.project_name,
                                                application_name=create_run_result.application_name,
                                                app_job_id=create_run_result.app_job_id,
                                                app_job_name=create_run_result.app_job_name)
        return run_start_result

    def list_run(self, platform: str, project_name: str, run_name: str, user_api_key: str,
                 run_id: str = None) -> FedMLRunModelList:
        run_list_result = None
        run_list_json = {
            "platformType": platform,
            "jobName": run_name if run_name is not None else "",
            "jobId": run_id if run_id is not None else "",
            "projectName": project_name if project_name is not None else "",
            "userApiKey": user_api_key
        }
        response = self._request(request_url=ServerConstants.get_run_list_url(),
                                 request_json=run_list_json,
                                 config_version=self.config_version)

        response_data = self._get_data_from_response(response=response)
        if response_data is not None and response_data.get("jobList", None) is not None:
            run_list_result = FedMLRunModelList(response_data)
        return run_list_result

    def stop_run(self, platform: str, user_api_key: str, run_id: str) -> bool:
        run_stop_json = {
            "platformType": platform,
            "jobId": run_id,
            "apiKey": user_api_key
        }
        response = self._request(request_url=ServerConstants.get_run_stop_url(),
                                 request_json=run_stop_json,
                                 config_version=self.config_version)
        response_data = self._get_data_from_response(response=response)
        return False if response_data is None else True

    def get_run_logs(self, run_id: str, page_num: int, page_size: int, user_api_key: str) -> FedMLRunLogModelList:
        run_log_list_result = None
        run_logs_json = {
            "apiKey": user_api_key,
            "edgeId": "-1",
            "pageNum": page_num,
            "pageSize": page_size,
            "runId": run_id,
            "timeZone": Constants.get_current_time_zone()
        }
        response = self._request(request_url=ServerConstants.get_run_logs_url(),
                                 request_json=run_logs_json,
                                 config_version=self.config_version)
        response_data = self._get_data_from_response(response=response)
        if response_data is not None:
            run_log_list_result = FedMLRunLogModelList(response_data)
        return run_log_list_result

    def _get_run_create_json(self, platform: str, project_name: str, application_name: str,
                             device_server: str, device_edges: List[str], api_key: str, task_type: str, app_job_id: str,
                             app_job_name: str, cluster: str = None, config_id: str = None,
                             feature_entry_point: FeatureEntryPoint = None):

        if not (device_server and device_edges):
            device_lists = [{"account": "", "edgeIds": [], "serverId": 0}]
        else:
            device_lists = list()
            device_item = dict()
            device_item["account"] = 0
            device_item["serverId"] = device_server
            device_item["edgeIds"] = str(device_edges).split(',')
            device_lists.append(device_item)
        run_create_json = {
            "platformType": platform,
            "name": "",
            "applicationName": application_name,
            "devices": device_lists,
            "urls": [],
            "apiKey": api_key,
            "needConfirmation": True,
        }

        if cluster:
            run_create_json["clusterName"] = cluster

        if project_name and len(str(project_name).strip(' ')) > 0:
            run_create_json["projectName"] = project_name
        else:
            run_create_json["projectName"] = ""

        if task_type:
            run_create_json["jobType"] = task_type

        if platform == "octopus":
            if project_name and len(str(project_name).strip(' ')) > 0:
                run_create_json["projectName"] = project_name
            else:
                run_create_json["projectName"] = "Cheetah_HelloWorld"
            run_create_json["name"] = str(uuid.uuid4())

        if app_job_id:
            run_create_json["applicationId"] = app_job_id

        if config_id:
            run_create_json["applicationConfigId"] = config_id

        if app_job_name:
            run_create_json["applicationName"] = app_job_name

        if feature_entry_point is not None:
            run_create_json["featureEntryPoint"] = feature_entry_point.value

        return run_create_json

    def _get_run_start_json(self, run_id: str, platform: str, project_name: str, application_name: str,
                            device_server: str, device_edges: List[str], api_key: str,
                            task_type: str, app_job_id: str, app_job_name: str,
                            feature_entry_point: FeatureEntryPoint = None):
        run_start_json = self._get_run_create_json(platform=platform, project_name=project_name,
                                                   application_name=application_name,
                                                   device_server=device_server, device_edges=device_edges,
                                                   api_key=api_key, task_type=task_type,
                                                   app_job_id=app_job_id, app_job_name=app_job_name,
                                                   feature_entry_point=feature_entry_point)
        run_start_json["jobId"] = run_id
        run_start_json["needConfirmation"] = False
        return run_start_json

    @staticmethod
    def _request(request_url: str, request_json: dict, config_version: str) -> requests.Response:
        request_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        args = {"config_version": config_version}
        cert_path = MLOpsConfigs.get_instance(args).get_cert_path_with_version()
        if cert_path:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    request_url, verify=True, headers=request_headers, json=request_json
                )
            except requests.exceptions.SSLError:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    request_url, verify=True, headers=request_headers, json=request_json
                )
        else:
            response = requests.post(request_url, headers=request_headers, json=request_json)
        return response

    @staticmethod
    def _get_data_from_response(response: Response):
        if response.status_code != 200:
            return None
        else:
            resp_data = response.json()
            code = resp_data.get("code", None)
            data = resp_data.get("data", None)
            if code is None or data is None or code == "FAILURE":
                return resp_data
        return data
