import argparse
import os
import uuid

import requests

from ...core.common.singleton import Singleton
from ..server_deployment.server_constants import ServerConstants
from ..edge_deployment.client_constants import ClientConstants
from ...core.mlops.mlops_configs import MLOpsConfigs
from ...core.distributed.communication.s3.remote_storage import S3Storage


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
        client_package_url = self.push_app_package_to_s3(application_name, client_package_file, user_id) \
            if client_package_file != "" else ""
        server_package_url = self.push_app_package_to_s3(application_name, server_package_file, user_id) \
            if client_package_file != "" else ""

        result = self.create_app_api(platform, application_name,
                                     client_package_url, client_package_file, server_package_url, server_package_file,
                                     user_id, user_api_key)
        if result is None:
            return False

        return True

    def create_app_api(self, platform, application_name,
                       user_id, user_api_key,
                       client_package_file=None, server_package_file=None):
        app_create_result = None

        if client_package_file is None and server_package_file is None:
            return False

        client_package_url = self.push_app_package_to_s3(application_name, client_package_file, user_id) \
            if client_package_file is not None else ""
        server_package_url = self.push_app_package_to_s3(application_name, server_package_file, user_id) \
            if server_package_file is not None else ""

        app_create_url = ServerConstants.get_app_create_url(self.config_version)
        app_create_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        app_create_json = {
            "owner": user_id,
            "avatar": "https://fedml.s3.us-west-1.amazonaws.com/profile_picture2.png",
            "githubLink": "",
            "accessPermission": 1,
            "applicationName": application_name,
            "privateLocalData": "",
            "pictureUrl": "",
            "platformId": "1",
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
            "parameter": {
                "common_args": {
                    "training_type": "cross_silo",
                    "scenario": "horizontal",
                    "using_mlops": False,
                    "random_seed": 0
                },
                "environment_args": {
                    "bootstrap": "config/bootstrap.sh"
                },
            },
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
            "fileList": [
                {
                    "fileName": server_package_file,
                    "fileDescribe": "",
                    "fileUrl": server_package_url,
                    "isFolder": 0,
                    "type": 1
                },
                {
                    "fileName": client_package_file,
                    "fileDescribe": "",
                    "fileUrl": client_package_url,
                    "isFolder": 0,
                    "type": 2
                },
                {
                    "fileName": "",
                    "fileDescribe": "",
                    "fileUrl": "",
                    "isFolder": 0,
                    "type": 3
                }
            ],
            "applicationConfigList": [],
            "userId": user_id,
            "apiKey": user_api_key
        }

        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    app_create_url, verify=True, headers=app_create_api_headers, json=app_create_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    app_create_url, verify=True, headers=app_create_api_headers, json=app_create_json
                )
        else:
            response = requests.post(app_create_url, headers=app_create_api_headers, json=app_create_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            app_create_result = resp_data

        return app_create_result

    def update_app(self, platform, application_name,
                   user_id, user_api_key,
                   client_package_file=None, server_package_file=None):
        if client_package_file is None and server_package_file is None:
            return False

        client_package_url = self.push_app_package_to_s3(application_name, client_package_file, user_id) \
            if client_package_file is not None else ""
        server_package_url = self.push_app_package_to_s3(application_name, server_package_file, user_id) \
            if server_package_file is not None else ""

        result = self.update_app_api(platform, application_name,
                                     client_package_url, client_package_file, server_package_url, server_package_file,
                                     user_id, user_api_key)
        if result is None:
            return False

        return True

    def update_app_api(self, platform, application_name,
                       client_package_url, client_package_file, server_package_url, server_package_file,
                       user_id, user_api_key):
        app_create_result = None
        app_create_url = ServerConstants.get_app_update_url(self.config_version)
        app_create_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        app_create_json = {
            "owner": user_id,
            "avatar": "https://fedml.s3.us-west-1.amazonaws.com/profile_picture2.png",
            "githubLink": "",
            "accessPermission": 1,
            "applicationName": application_name,
            "privateLocalData": "",
            "pictureUrl": "",
            "platformId": "1",
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
            "parameter": {
                "common_args": {
                    "training_type": "cross_silo",
                    "scenario": "horizontal",
                    "using_mlops": False,
                    "random_seed": 0
                },
                "environment_args": {
                    "bootstrap": "config/bootstrap.sh"
                },
            },
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
            "fileList": [
                {
                    "fileName": server_package_file,
                    "fileDescribe": "",
                    "fileUrl": server_package_url,
                    "isFolder": 0,
                    "type": 1
                },
                {
                    "fileName": client_package_file,
                    "fileDescribe": "",
                    "fileUrl": client_package_url,
                    "isFolder": 0,
                    "type": 2
                },
                {
                    "fileName": "",
                    "fileDescribe": "",
                    "fileUrl": "",
                    "isFolder": 0,
                    "type": 3
                }
            ],
            "applicationConfigList": [],
            "userId": user_id,
            "apiKey": user_api_key
        }
        args = {"config_version": self.config_version}
        _, cert_path = MLOpsConfigs.get_instance(args).get_request_params_with_version(self.config_version)
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    app_create_url, verify=True, headers=app_create_api_headers, json=app_create_json
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    app_create_url, verify=True, headers=app_create_api_headers, json=app_create_json
                )
        else:
            response = requests.post(app_create_url, headers=app_create_api_headers, json=app_create_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            app_create_result = resp_data

        return app_create_result

    def push_app_package_to_s3(self, app_name, app_package_path, user_id):
        args = {"config_version": self.config_version}
        _, s3_config = MLOpsConfigs.get_instance(args).fetch_configs(self.config_version)
        s3_storage = S3Storage(s3_config)
        app_dst_key = "{}@{}@{}".format(user_id, app_name, str(uuid.uuid4()))
        app_storage_url = s3_storage.upload_file_with_progress(app_package_path, app_dst_key)
        return app_storage_url

    def pull_app_package_from_s3(self, model_storage_url, model_name):
        args = {"config_version": self.config_version}
        _, s3_config = MLOpsConfigs.get_instance(args).fetch_configs(self.config_version)
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