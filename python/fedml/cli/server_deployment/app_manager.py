import argparse
import requests

from ...core.common.singleton import Singleton
from .server_constants import ServerConstants
from ...core.mlops.mlops_configs import MLOpsConfigs


class FedMLAppManager(Singleton):

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLAppManager()

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def create_app(self, platform, application_name,
                   client_package_url, client_package_file, server_package_url, server_package_file,
                   user_id, user_api_key):
        result = self.create_app_api(platform, application_name,
                                     client_package_url, client_package_file, server_package_url, server_package_file,
                                     user_id, user_api_key)
        if result is None:
            return False

        return True

    def create_app_api(self, platform, application_name,
                       client_package_url, client_package_file, server_package_url, server_package_file,
                       user_id, user_api_key):
        app_create_result = None
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
                   client_package_url, client_package_file, server_package_url, server_package_file,
                   user_id, user_api_key):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cf", "-c", help="config file")
    parser.add_argument("--role", "-r", type=str, default="client", help="role")
    in_args = parser.parse_args()
