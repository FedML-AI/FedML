import json
import time
import uuid

import requests
from fedml.computing.scheduler.scheduler_entry.constants import Constants

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.core.mlops.mlops_configs import MLOpsConfigs


class FedMLJobManager(Singleton):

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLJobManager()

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def start_job(self, platform, project_name, application_name, device_server, device_edges,
                  user_api_key, no_confirmation=False, job_id=None):
        return self.start_job_api(platform, project_name, application_name, device_server, device_edges,
                                  user_api_key, no_confirmation=no_confirmation, job_id=job_id)

    def start_job_api(self, platform, project_name, application_name, device_server, device_edges,
                      user_api_key, no_confirmation=False, job_id=None):
        job_start_result = None
        jot_start_url = ServerConstants.get_job_start_url(self.config_version)
        job_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        if device_server == "" or device_edges == "":
            device_lists = [{"account": "", "edgeIds": [], "serverId": 0}]
        else:
            device_lists = list()
            device_item = dict()
            device_item["account"] = 0
            device_item["serverId"] = device_server
            device_item["edgeIds"] = str(device_edges).split(',')
            device_lists.append(device_item)
        job_start_json = {
            "platformType": platform,
            "name": "",
            "applicationName": application_name,
            "applicationConfigId": 0,
            "devices": device_lists,
            "urls": [],
            "apiKey": user_api_key,
            "needConfirmation": True if user_api_key is None or user_api_key == "" else not no_confirmation
        }
        if project_name is not None and len(str(project_name).strip(' ')) > 0:
            job_start_json["projectName"] = project_name
        else:
            job_start_json["projectName"] = ""

        if platform == "octopus":
            if project_name is not None and len(str(project_name).strip(' ')) > 0:
                job_start_json["projectName"] = project_name
            else:
                job_start_json["projectName"] = "Cheetah_HelloWorld"
            job_start_json["name"] = str(uuid.uuid4())

        if job_id is not None:
            job_start_json["jobId"] = job_id
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
            print(f"Launch job with response.status_code = {response.status_code}, response.content: {response.content}")
            pass
        else:
            resp_data = response.json()
            job_start_result = FedMLJobStartedModel(resp_data["data"], response=resp_data)

        return job_start_result

    def list_job(self, platform, project_name, job_name, user_api_key, job_id=None):
        return self.list_job_api(platform, project_name, job_name, user_api_key, job_id=job_id)

    def list_job_api(self, platform, project_name, job_name, user_api_key, job_id=None):
        job_list_result = None
        jot_list_url = ServerConstants.get_job_list_url(self.config_version)
        job_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        job_list_json = {
            "platformType": platform,
            "jobName": job_name if job_name is not None else "",
            "projectName": project_name if project_name is not None else "",
            "userApiKey": user_api_key
        }
        if job_id is not None and job_id != "":
            job_list_json["jobId"] = job_id
        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    jot_list_url, verify=True, headers=job_api_headers, json=job_list_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    jot_list_url, verify=True, headers=job_api_headers, json=job_list_json
                )
        else:
            response = requests.post(jot_list_url, headers=job_api_headers, json=job_list_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                return None
            job_list_result = FedMLJobModelList(resp_data["data"])

        return job_list_result

    def stop_job(self, platform, job_id, user_api_key):
        return self.stop_job_api(platform, job_id, user_api_key)

    def stop_job_api(self, platform, job_id, user_api_key):
        jot_stop_url = ServerConstants.get_job_stop_url(self.config_version)
        job_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        job_stop_json = {
            "platformType": platform,
            "jobId": job_id,
            "apiKey": user_api_key
        }
        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    jot_stop_url, verify=True, headers=job_api_headers, json=job_stop_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    jot_stop_url, verify=True, headers=job_api_headers, json=job_stop_json
                )
        else:
            response = requests.post(jot_stop_url, headers=job_api_headers, json=job_stop_json)
        if response.status_code != 200:
            return False
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                return False

        return True

    def get_job_logs(self, job_id, page_num, page_size, user_api_key):
        return self.get_job_logs_api(job_id, page_num, page_size, user_api_key)

    def get_job_logs_api(self, job_id, page_num, page_size,user_api_key):
        job_log_list_result = None
        jot_logs_url = ServerConstants.get_job_logs_url(self.config_version)
        job_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        job_logs_json = {
            "apiKey": user_api_key,
            "edgeId": "-1",
            "pageNum": page_num,
            "pageSize": page_size,
            "runId": job_id,
            "timeZone": Constants.get_current_time_zone()
        }
        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    jot_logs_url, verify=True, headers=job_api_headers, json=job_logs_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    jot_logs_url, verify=True, headers=job_api_headers, json=job_logs_json
                )
        else:
            response = requests.post(jot_logs_url, headers=job_api_headers, json=job_logs_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                return None
            job_log_list_result = FedMLJobLogModelList(resp_data["data"])

        return job_log_list_result


class FedMLJobStartedModel(object):
    def __init__(self, job_started_json, job_name=None, response=None):
        if isinstance(job_started_json, dict):
            self.job_id = job_started_json.get("job_id", "0")
            self.job_name = job_started_json.get("job_name", job_name)
            self.status = job_started_json.get("status", Constants.MLOPS_CLIENT_STATUS_NOT_STARTED)
            self.job_url = job_started_json.get("job_url", job_started_json)
            self.gpu_matched = list()
            self.message = job_started_json.get("message", None)
            gpu_list_json = job_started_json.get("gpu_matched", None)
            if gpu_list_json is not None:
                for gpu_dev_json in gpu_list_json:
                    self.gpu_matched.append(FedMLGpuDevices(gpu_dev_json))
            self.started_time = job_started_json.get("started_time", time.time())
        else:
            self.job_id = "0"
            self.job_name = job_name
            self.status = response.get("code")
            self.job_url = job_started_json
            self.started_time = time.time()
            self.message = response.get("message")


class FedMLGpuDevices(object):
    def __init__(self, gpu_device_json):
        self.gpu_vendor = gpu_device_json.get("gpu_vendor", None)
        self.gpu_num = gpu_device_json.get("total_gpu_count", None)
        self.gpu_type = gpu_device_json.get("gpu_type", None)
        self.cost = gpu_device_json.get("cost", None)
        self.mem_size = gpu_device_json.get("gpu_mem", None)
        self.gpu_region = gpu_device_json.get("gpu_region", "DEFAULT")
        self.gpu_region = "DEFAULT" if self.gpu_region is None or self.gpu_region == "" else self.gpu_region
        self.cpu_count = gpu_device_json.get("cpu_count", None)
        self.cpu_count = None if self.cpu_count is not None and int(self.cpu_count) <= 0 else self.cpu_count
        self.gpu_count = gpu_device_json.get("got_gpu_count", None)
        self.gpu_name = gpu_device_json.get("gpu_name", None)
        self.gpu_instance = self.gpu_name
        self.gpu_provider = gpu_device_json.get("gpu_provider", None)


class FedMLJobModelList(object):
    def __init__(self, job_list_json):
        job_list_data = job_list_json["jobList"]
        self.job_list = list()
        for job_obj_json in job_list_data:
            job_obj = FedMLJobModel(job_obj_json)
            self.job_list.append(job_obj)


class FedMLJobModel(object):
    def __init__(self, job_json):
        self.job_id = job_json["id"]
        self.job_name = job_json["name"]
        self.status = job_json["status"]
        self.started_time = Constants.format_time_trimmed_tz(job_json.get("createTime", "0"))
        self.ended_time = Constants.format_time_trimmed_tz(job_json.get("endTime", "0"))
        self.running_time = job_json.get("spendTime", 0)
        self.compute_start_time = self.started_time
        self.compute_end_time = self.ended_time
        self.compute_duration = job_json.get("spendTime", 0)
        if self.compute_duration is not None and self.compute_duration != 'None':
            self.compute_duration = self.compute_duration / Constants.TIME_PER_HOUR_TO_MS
            self.compute_duration = round(self.compute_duration, 4)
        self.cost = job_json.get("cost", 0.0)
        self.job_url = job_json.get("jobUrl", "")
        gpu_machines = job_json.get("gpuMachines", None)
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

    def parse(self, job_json):
        pass


class FedMLJobLogModelList(object):
    def __init__(self, job_log_list_json):
        self.log_full_url = job_log_list_json.get("log_full_url", None)
        self.log_full_url = None if self.log_full_url is not None and self.log_full_url == "" else self.log_full_url
        log_devices_json = job_log_list_json.get("devices", [])
        self.log_devices = list()
        for log_dev in log_devices_json:
            self.log_devices.append(FedMLJobLogDeviceModel(log_dev))
        self.total_num = job_log_list_json.get("total_num", 0)
        self.total_pages = job_log_list_json.get("total_pages", 0)
        self.current_page = job_log_list_json.get("current_page", 0)
        self.log_lines = job_log_list_json.get("logs", [])


class FedMLJobLogDeviceModel(object):
    def __init__(self, jog_log_device_json):
        self.log_url = jog_log_device_json.get("log_url", None)
        self.log_url = None if self.log_url is not None and self.log_url == "" else self.log_url
        self.device_name = jog_log_device_json.get("name", None)
        self.device_id = jog_log_device_json.get("id", None)
