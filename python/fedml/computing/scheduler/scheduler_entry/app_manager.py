import argparse
import os
import uuid

import requests

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.mlops.mlops_configs import MLOpsConfigs
from fedml.core.distributed.communication.s3.remote_storage import S3Storage


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
                                                               "Submitting your job to FedML® Launch platform")
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