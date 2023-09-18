import argparse
import os
import uuid

import requests

from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
from fedml.computing.scheduler.model_scheduler import device_client_constants
from fedml.computing.scheduler.scheduler_entry.constants import Constants

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.mlops.mlops_configs import MLOpsConfigs
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards


class FedMLAppManager(Singleton):

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLAppManager()

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def create_app(self, platform, application_name, client_package_file, server_package_file,
                   user_id, user_api_key):
        return self.update_app(platform, application_name, None, user_id, user_id, user_api_key,
                               client_package_file, server_package_file)

    def update_app(self, platform, application_name, app_config,
                   user_api_key, client_package_file=None, server_package_file=None):
        if client_package_file is None and server_package_file is None:
            return False

        client_package_url = self.push_app_package_to_s3(application_name, client_package_file) \
            if client_package_file is not None else None
        server_package_url = self.push_app_package_to_s3(application_name, server_package_file) \
            if server_package_file is not None else None

        result = self.update_app_api(platform, application_name, app_config,
                                     client_package_url,
                                     os.path.basename(client_package_file) if client_package_file is not None else None,
                                     server_package_url,
                                     os.path.basename(server_package_file) if server_package_file is not None else None,
                                     user_api_key)
        if result is None:
            return False

        return True

    def update_app_api(self, platform, application_name, app_config,
                       client_package_url, client_package_file, server_package_url, server_package_file,
                       user_api_key):
        app_update_result = None
        app_update_url = ServerConstants.get_app_update_url(self.config_version)
        app_update_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        app_update_json = {
            "avatar": "https://fedml.s3.us-west-1.amazonaws.com/profile_picture2.png",
            "githubLink": "",
            "accessPermission": 1,
            "applicationName": application_name,
            "privateLocalData": "",
            "pictureUrl": "",
            "platformId": platform,
            "dataType": 1,
            "dataId": 1,
            "description": "# Please describe your application with this markdown editor\n"
                           "To make it easier to understand and use this application, "
                           "we suggest describing the following information:\n"
                           "1. A general scenario description for this application.\n"
                           "2. The ML task definition of this application, including the input and output.\n"
                           "3. The dataset format\n"
                           "4. A high-level deep/machine learning model definition\n"
                           "5. Some other tips for your application users.",
            "parameter": {},
            "tagList": [
                {
                    "tagId": 120,
                    "tagName": "Industry IoT",
                    "isRoot": 0,
                    "parentId": 1,
                    "count": 163
                },
                {
                    "tagId": 124,
                    "tagName": "Computer Vision",
                    "isRoot": 0,
                    "parentId": 3,
                    "count": 190
                }
            ],
            "fileList": [],
            "applicationConfigList": [],
            "apiKey": user_api_key
        }

        if app_config is not None:
            app_update_json["parameter"] = app_config

        package_file_list = list()
        if server_package_url is not None:
            package_file_list.append({
                "fileName": server_package_file,
                "fileDescribe": "",
                "fileUrl": server_package_url,
                "isFolder": 0,
                "type": 1})
        else:
            package_file_list.append({
                "fileName": "",
                "fileDescribe": "",
                "fileUrl": "",
                "isFolder": 0,
                "type": 1})
        if client_package_url is not None:
            package_file_list.append({
                "fileName": client_package_file,
                "fileDescribe": "",
                "fileUrl": client_package_url,
                "isFolder": 0,
                "type": 2})
        else:
            package_file_list.append({
                "fileName": "",
                "fileDescribe": "",
                "fileUrl": "",
                "isFolder": 0,
                "type": 2})
        app_update_json["fileList"] = package_file_list

        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    app_update_url, verify=True, headers=app_update_api_headers, json=app_update_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    app_update_url, verify=True, headers=app_update_api_headers, json=app_update_json
                )
        else:
            response = requests.post(app_update_url, headers=app_update_api_headers, json=app_update_json)
        if response.status_code != 200:
            print(f"Update application with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            app_update_result = resp_data

        return app_update_result

    def push_app_package_to_s3(self, app_name, app_package_path):
        args = {"config_version": self.config_version}
        _, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
        s3_storage = S3Storage(s3_config)
        app_dst_key = "{}@{}".format(app_name, str(uuid.uuid4()))
        app_storage_url = s3_storage.upload_file_with_progress(app_package_path, app_dst_key,
                                                               out_progress_to_err=False,
                                                               progress_desc="Submitting your job to "
                                                                             "FedML® Launch platform")
        return app_storage_url

    def pull_app_package_from_s3(self, model_storage_url, model_name):
        args = {"config_version": self.config_version}
        _, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
        s3_storage = S3Storage(s3_config)
        local_app_package = os.path.join(ClientConstants.get_package_download_dir(), model_name)
        local_app_package = "{}.zip".format(local_app_package)
        print("Pulling......")
        ClientConstants.retrieve_and_unzip_package(model_storage_url,
                                                   model_name,
                                                   local_app_package,
                                                   ClientConstants.get_model_dir())
        if os.path.exists(local_app_package):
            return local_app_package

        return ""

    def build_model(self, model_name, workspace_dir):
        FedMLModelCards.get_instance().set_config_version(self.config_version)
        FedMLModelCards.get_instance().delete_model(model_name)
        if not FedMLModelCards.get_instance().create_model(model_name):
            return Constants.ERROR_CODE_MODEL_CREATE_FAILED, None

        if not FedMLModelCards.get_instance().add_model_files(model_name, workspace_dir):
            return Constants.ERROR_CODE_MODEL_ADD_FILES_FAILED, None

        model_zip_path = FedMLModelCards.get_instance().build_model(model_name)
        if model_zip_path is None or model_zip_path == "":
            return Constants.ERROR_CODE_MODEL_BUILD_FAILED, None

        return 0, model_zip_path

    def push_model_to_s3(self, model_name, model_zip_path):
        FedMLModelCards.get_instance().set_config_version(self.config_version)
        return FedMLModelCards.get_instance().push_model_to_s3(
            model_name, model_zip_path, "FedMLLaunchServe",
            progress_desc="Submitting your job to FedML® Launch platform")

    def check_model_package(self, workspace):
        model_config_file = os.path.join(
            workspace, device_client_constants.ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        if not os.path.exists(model_config_file):
            return False

        try:
            model_yaml = load_yaml_config(model_config_file)
        except Exception as e:
            return False

        return True

    def check_model_exists(self, model_name, api_key):
        FedMLModelCards.get_instance().set_config_version(self.config_version)
        result = FedMLModelCards.get_instance().list_models(model_name, user_id="", user_api_key=api_key)
        if result is not None and len(result.model_list) > 0:
            return True

        return False

    def update_model(self, model_name, workspace, api_key):
        FedMLModelCards.get_instance().set_config_version(self.config_version)

        error_code, model_zip_path = self.build_model(model_name, workspace)
        if error_code != 0:
            return None

        model_storage_url = self.push_model_to_s3(model_name, model_zip_path)
        if model_storage_url == "":
            return None

        model_dir = os.path.join(device_client_constants.ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            return None
        model_config_file = os.path.join(
            model_dir, device_client_constants.ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        model_yaml = load_yaml_config(model_config_file)

        upload_result = FedMLModelCards.get_instance().upload_model_api(model_name, model_yaml, model_storage_url,
                                                                        None, "", api_key,
                                                                        is_from_open=False,
                                                                        local_server=None)
        if upload_result is None:
            return None

        result = FedMLModelUploadResult(model_name, model_storage_url=model_storage_url)

        return result


class FedMLModelUploadResult(object):
    def __init__(self, model_name, model_version="", model_storage_url="", endpoint_name=""):
        self.model_name = model_name
        self.model_version = model_version
        self.model_storage_url = model_storage_url
        self.endpoint_name = endpoint_name
