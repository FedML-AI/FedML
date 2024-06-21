import base64
import json
import logging
import os
import platform
import traceback

import setproctitle

import fedml
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program
from fedml.computing.scheduler.scheduler_core.account_manager import FedMLAccountManager


class FedMLCloudServerManager:
    FEDML_CLOUD_SERVER_PREFIX = "fedml-server-run-"
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    STATUS_IDLE = "IDLE"
    FEDML_SERVER_BASE_IMAGE = "/fedml-device-image:"

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
        self.server_docker_base_image = FedMLCloudServerManager._get_server_base_image(image_version)
        self.cloud_server_name = None

    @staticmethod
    def start_local_cloud_server(user, api_key, os_name, version, cloud_device_id, runner_cmd_encoded):
        if platform.system() != "Windows":
            os.setsid()

        print(f"start cloud server, device id {cloud_device_id}, runner cmd {runner_cmd_encoded}")
        pip_source_dir = os.path.dirname(__file__)
        login_cmd = os.path.join(pip_source_dir, "server_login.py")
        run_cmd = f"{get_python_program()} -W ignore {login_cmd} -t login -r cloud_server -u {str(user)} " \
                  f"-k {api_key} -v {version} -id {cloud_device_id} -rc {runner_cmd_encoded}"
        os.system(run_cmd)

    def start_local_master_server(
            self, user, api_key, os_name, version, cloud_device_id, run_id, payload,
            communication_manager=None, sender_message_queue=None, status_center_queue=None,
            master_agent_instance=None, process_name=None
    ):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        logging.info(f"Local master server pid: {os.getpid()}")
        if platform.system() != "Windows":
            os.setsid()

        master_agent_instance.login(
            user, api_key=api_key, device_id=cloud_device_id, os_name=os_name,
            role=FedMLAccountManager.ROLE_CLOUD_SERVER, runner_cmd=payload,
            communication_manager=None, sender_message_queue=None,
            status_center_queue=None)

        master_agent_instance.stop()

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

    @staticmethod
    def stop_cloud_server(run_id, server_id, agent_config):
        cloud_server_name = FedMLCloudServerManager._get_cloud_server_name(run_id, server_id)
        server_docker_image = (
                agent_config["docker_config"]["registry_server"]
                + agent_config["docker_config"]["registry_dir"]
                + FedMLCloudServerManager._get_server_base_image(fedml.get_env_version())
        )
        delete_deployment_cmd = (
                "export FEDML_AGGREGATOR_NAME="
                + cloud_server_name
                + ";export FEDML_AGGREGATOR_SVC="
                + cloud_server_name
                + ";export FEDML_AGGREGATOR_VERSION="
                + fedml.get_env_version()
                + ';export FEDML_AGGREGATOR_IMAGE_PATH="'
                + server_docker_image
                + '"'
                + ";export FEDML_CONF_ID="
                + cloud_server_name
                + ";export FEDML_DATA_PV_ID="
                + cloud_server_name
                + ";export FEDML_DATA_PVC_ID="
                + cloud_server_name
                + ";export FEDML_REGISTRY_SECRET_SUFFIX="
                + cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + fedml.get_env_version()
                + " delete deployment "
                + cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + fedml.get_env_version()
                + " delete svc "
                + cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + fedml.get_env_version()
                + " delete secret secret-"
                + cloud_server_name
        )
        logging.info("stop run with k8s: " + delete_deployment_cmd)
        os.system(delete_deployment_cmd)

    @staticmethod
    def _get_server_base_image(version):
        return f"{FedMLCloudServerManager.FEDML_SERVER_BASE_IMAGE}{version}"

    @staticmethod
    def _get_cloud_server_name(run_id, server_id):
        return f"{FedMLCloudServerManager.FEDML_CLOUD_SERVER_PREFIX}{run_id}-{server_id}"
