
import os
import time
import uuid
import fedml

import requests

from fedml.computing.scheduler.comm_utils.security_utils import get_content_hash, get_file_hash
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
from fedml.computing.scheduler.model_scheduler import device_client_constants
from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.scheduler_entry.launch_app_interface import FedMLLaunchAppDataInterface

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.mlops.mlops_configs import MLOpsConfigs
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards


class FedMLAppManager(Singleton):

    def __init__(self):
        self.config_version = fedml.get_env_version()

    @staticmethod
    def get_instance():
        return FedMLAppManager()

    def create_app(self, platform, application_name, client_package_file, server_package_file,
                   user_id, user_api_key):
        return self.update_app(platform, application_name, None, user_id, user_id, user_api_key,
                               client_package_file, server_package_file)

    def update_app(self, platform, application_name, app_config,
                   user_api_key, client_package_file=None, server_package_file=None,
                   workspace=None, model_name=None, model_version=None,
                   model_url=None, app_id=None, config_id=None,
                   job_type=None, job_subtype=None):
        if app_id is not None:
            return self.update_app_with_app_id_api(app_id, config_id, app_config, user_api_key)

        if client_package_file is None and server_package_file is None:
            return False

        # should_upload, workspace_hash, app_hash = self.should_upload_app_package(workspace)
        # if not should_upload:
        #     return True

        client_package_url = self.push_app_package_to_s3(application_name, client_package_file) \
            if client_package_file is not None else None
        server_package_url = self.push_app_package_to_s3(application_name, server_package_file) \
            if server_package_file is not None else None

        result = self.update_app_api(platform, application_name, app_config,
                                     client_package_url,
                                     os.path.basename(client_package_file) if client_package_file is not None else None,
                                     server_package_url,
                                     os.path.basename(server_package_file) if server_package_file is not None else None,
                                     user_api_key, job_type=job_type,  job_subtype=job_subtype)
        if result is None:
            return False

        # self.update_local_app_storage(
        #     application_name, app_config, workspace_hash, app_hash,
        #     client_package_url=client_package_url, server_package_url=server_package_url,
        #     client_package_file=client_package_file, server_package_file=server_package_file,
        #     workspace=workspace, model_name=model_name, model_version=model_version, model_url=model_url)

        return True

    def should_upload_app_package(self, workspace):
        workspace_hash = get_content_hash(workspace)
        app_hash = get_file_hash(workspace)
        app_obj = FedMLLaunchAppDataInterface.get_app_by_id(workspace_hash)
        return (False if app_obj is not None and str(app_obj.app_hash) == str(app_hash) else False,
                workspace_hash, app_hash)

    def update_local_app_storage(self, application_name, app_config,
                                 workspace_hash, app_hash,
                                 client_package_url=None, server_package_url=None,
                                 client_package_file=None, server_package_file=None,
                                 workspace=None, model_name=None, model_version=None,
                                 model_url=None):
        client_diff_url = ""
        client_diff_file = ""
        server_diff_url = ""
        server_diff_file = ""
        app_obj = FedMLLaunchAppDataInterface.get_app_by_id(workspace_hash)
        if app_obj is None:
            app_obj = FedMLLaunchAppDataInterface(
                workspace_hash, application_name, app_config, workspace, workspace_hash,
                app_hash, client_package_url, client_package_file, server_package_url, server_package_file,
                client_diff_url, client_diff_file, server_diff_url, server_diff_file,
                model_name, model_version, model_url, str(time.time()))
            FedMLLaunchAppDataInterface.insert_app_to_db(app_obj)
        else:
            app_obj.app_hash = app_hash
            app_obj.client_package_url = client_package_file if client_package_url is not None else \
                app_obj.client_package_url
            app_obj.client_package_file = client_package_file if client_package_url is not None else \
                app_obj.client_package_file
            app_obj.server_package_url = server_package_url if server_package_url is not None else \
                app_obj.server_package_url
            app_obj.server_package_file = server_package_file if server_package_file is not None else \
                app_obj.server_package_file
            app_obj.workspace = workspace if workspace is not None else app_obj.workspace
            app_obj.model_name = model_name if model_name is not None else app_obj.model_name
            app_obj.model_version = model_version if model_version is not None else app_obj.model_version
            app_obj.model_url = model_url if model_url is not None else app_obj.model_url
            FedMLLaunchAppDataInterface.update_app_to_db(app_obj)

    def update_app_api(self, platform, application_name, app_config,
                       client_package_url, client_package_file, server_package_url, server_package_file,
                       user_api_key, job_type=None, job_subtype=None):
        platform_id = Constants.platform_str_to_type(platform)
        app_update_result = None
        app_update_url = ServerConstants.get_app_update_url()
        app_update_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        app_update_json = {
            "avatar": "https://fedml.s3.us-west-1.amazonaws.com/profile_picture2.png",
            "githubLink": "",
            "accessPermission": 1,
            "applicationName": application_name,
            "privateLocalData": "",
            "pictureUrl": "",
            "platformId": platform_id,
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

        if job_type is not None:
            app_update_json["jobType"] = job_type

        if job_subtype is not None:
            app_update_json["jobSubType"] = job_subtype

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

        app_update_json

        args = {"config_version": self.config_version}
        cert_path = MLOpsConfigs.get_cert_path_with_version()
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

    def update_app_with_app_id_api(self, app_id, config_id, app_config, user_api_key):
        app_update_result = None
        app_update_url = ServerConstants.get_app_update_with_app_id_url()
        app_update_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        app_update_json = {
            "applicationConfig": app_config,
            "applicationId": app_id,
            "apiKey": user_api_key
        }

        if config_id is not None:
            app_update_json["applicationConfigId"] = config_id

        args = {"config_version": self.config_version}
        cert_path = MLOpsConfigs.get_cert_path_with_version()
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
            print(f"Update application using app id with response.status_code = {response.status_code}, "
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
        _, s3_config, _, _ = MLOpsConfigs.fetch_all_configs()
        s3_storage = S3Storage(s3_config)
        app_dst_key = "{}@{}".format(app_name, str(uuid.uuid4()))
        app_storage_url = s3_storage.upload_file_with_progress(app_package_path, app_dst_key,
                                                               out_progress_to_err=True,
                                                               progress_desc="Submitting your job to "
                                                                             "FedML® Nexus AI Platform")
        return app_storage_url

    def pull_app_package_from_s3(self, model_storage_url, model_name):
        args = {"config_version": self.config_version}
        _, s3_config, _, _ = MLOpsConfigs.fetch_all_configs()
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
        return FedMLModelCards.get_instance().push_model_to_s3(
            model_name, model_zip_path, "FedMLLaunchServe",
            show_progress=False,
            progress_desc="Submitting your job to FedML® Nexus AI Platform")

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
        result = FedMLModelCards.get_instance().list_models(model_name, api_key=api_key)
        return result

    def update_model(self, model_name, workspace, api_key, is_creating_model=True, model_object=None):
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

        if is_creating_model:
            upload_result = FedMLModelCards.get_instance().upload_model_api(model_name, model_yaml, model_storage_url,
                                                                            None, "", api_key,
                                                                            is_from_open=False)
        else:
            upload_result = FedMLModelCards.get_instance().update_model_api(
                model_name, model_yaml, model_storage_url, None, "", api_key, is_from_open=False,
                model_id=model_object.id)
        if upload_result is None:
            return None

        result = FedMLModelUploadResult(model_name, model_storage_url=model_storage_url)

        return result

    def delete_endpoint(self, api_key, endpoint_id):
        FedMLModelCards.get_instance().delete_endpoint_api(api_key, endpoint_id)


class FedMLModelUploadResult(object):
    def __init__(self, model_name, model_id="", model_version="", model_storage_url="", endpoint_name=""):
        self.model_id = model_id
        self.model_name = model_name
        self.model_version = model_version
        self.model_storage_url = model_storage_url
        self.endpoint_name = endpoint_name
