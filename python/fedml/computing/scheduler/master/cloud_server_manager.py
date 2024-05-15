import base64
import json
import logging
import os
import traceback

import fedml
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program


class FedMLCloudServerManager:
    FEDML_CLOUD_SERVER_PREFIX = "fedml-server-run-"
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    STATUS_IDLE = "IDLE"

    def __init__(self, args, run_id=None, edge_id=None, request_json=None, agent_config=None, version=None):
        self.server_docker_image = None
        self.args = args
        self.run_id = run_id
        self.edge_id = edge_id
        self.request_json = request_json
        self.agent_config = agent_config
        if version is None:
            version = fedml.get_env_version()
        self.version = version
        image_version = self.version
        if image_version == "local":
            image_version = "test"
        self.server_docker_base_image = "/fedml-device-image:" + image_version
        self.cloud_server_name = None

    @staticmethod
    def start_local_cloud_server(user, version, cloud_device_id, runner_cmd_encoded):
        print(f"start cloud server, device id {cloud_device_id}, runner cmd {runner_cmd_encoded}")
        pip_source_dir = os.path.dirname(__file__)
        login_cmd = os.path.join(pip_source_dir, "server_login.py")
        run_cmd = f"{get_python_program()} -W ignore {login_cmd} -t login -r cloud_server -u {str(user)} " \
                  f"-v {version} -id {cloud_device_id} -rc {runner_cmd_encoded}"
        os.system(run_cmd)

    def start_cloud_server_process_entry(self):
        try:
            self.start_cloud_server_process()
        except Exception as e:
            logging.info(f"Failed to start the cloud server. {traceback.format_exc()}")

    def start_cloud_server_process(self):
        run_config = self.request_json["run_config"]
        packages_config = run_config["packages_config"]
        self.start_cloud_server(packages_config)

    def start_cloud_server(self, packages_config):
        server_id = self.request_json["server_id"]
        self.cloud_server_name = f"{FedMLCloudServerManager.FEDML_CLOUD_SERVER_PREFIX}{self.run_id}-{server_id}"
        self.server_docker_image = (
                self.agent_config["docker_config"]["registry_server"]
                + self.agent_config["docker_config"]["registry_dir"]
                + self.server_docker_base_image
        )

        logging.info("docker image {}".format(self.server_docker_image))
        # logging.info("file_sys_driver {}".format(self.agent_config["docker_config"]["file_sys_driver"]))

        registry_secret_cmd = (
                "kubectl create namespace fedml-devops-aggregator-"
                + self.version
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete secret secret-"
                + self.cloud_server_name
                + " ;kubectl create secret docker-registry secret-"
                + self.cloud_server_name
                + " --docker-server="
                + self.agent_config["docker_config"]["registry_server"]
                + " --docker-username="
                + self.agent_config["docker_config"]["user_name"]
                + " --docker-password=$(aws ecr-public get-login-password --region "
                + self.agent_config["docker_config"]["public_cloud_region"]
                + ")"
                + " --docker-email=fedml@fedml.ai -n fedml-devops-aggregator-"
                + self.version
        )
        logging.info("Create secret cmd: " + registry_secret_cmd)
        os.system(registry_secret_cmd)

        message_bytes = json.dumps(self.request_json).encode("ascii")
        base64_bytes = base64.b64encode(message_bytes)
        runner_cmd_encoded = base64_bytes.decode("ascii")
        logging.info("runner_cmd_encoded: {}".format(runner_cmd_encoded))
        # logging.info("runner_cmd_decoded: {}".format(base64.b64decode(runner_cmd_encoded).decode()))
        cur_dir = os.path.dirname(__file__)
        run_deployment_cmd = (
                "export FEDML_AGGREGATOR_NAME="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_SVC="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_VERSION="
                + self.version
                + ';export FEDML_AGGREGATOR_IMAGE_PATH="'
                + self.server_docker_image
                + '"'
                + ";export FEDML_CONF_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PV_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PVC_ID="
                + self.cloud_server_name
                + ";export FEDML_REGISTRY_SECRET_SUFFIX="
                + self.cloud_server_name
                + ";export FEDML_ACCOUNT_ID=0"
                + ";export FEDML_SERVER_DEVICE_ID="
                + self.request_json.get("cloudServerDeviceId", "0")
                + ";export FEDML_VERSION="
                + self.version
                + ";export FEDML_PACKAGE_NAME="
                + packages_config.get("server", "")
                + ";export FEDML_PACKAGE_URL="
                + packages_config.get("serverUrl", "")
                + ";export FEDML_RUNNER_CMD="
                + runner_cmd_encoded
                + ";envsubst < "
                + os.path.join(cur_dir, "templates", "fedml-server-deployment.yaml")
                + " | kubectl apply -f - "
        )
        logging.info("start run with k8s: " + run_deployment_cmd)
        os.system(run_deployment_cmd)

    def stop_cloud_server(self):
        self.cloud_server_name = FedMLCloudServerManager.FEDML_CLOUD_SERVER_PREFIX + str(self.run_id) \
                                 + "-" + str(self.edge_id)
        self.server_docker_image = (
                self.agent_config["docker_config"]["registry_server"]
                + self.agent_config["docker_config"]["registry_dir"]
                + self.server_docker_base_image
        )
        delete_deployment_cmd = (
                "export FEDML_AGGREGATOR_NAME="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_SVC="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_VERSION="
                + self.version
                + ';export FEDML_AGGREGATOR_IMAGE_PATH="'
                + self.server_docker_image
                + '"'
                + ";export FEDML_CONF_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PV_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PVC_ID="
                + self.cloud_server_name
                + ";export FEDML_REGISTRY_SECRET_SUFFIX="
                + self.cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete deployment "
                + self.cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete svc "
                + self.cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete secret secret-"
                + self.cloud_server_name
        )
        logging.info("stop run with k8s: " + delete_deployment_cmd)
        os.system(delete_deployment_cmd)
