import logging

import docker

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.core.common.singleton import Singleton


class ContainerUtils(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return ContainerUtils()

    def get_container_logs(self, container_name):
        try:
            client = docker.from_env()
        except Exception:
            logging.error("Failed to connect to the docker daemon, please ensure that you have "
                          "installed Docker Desktop or Docker Engine, and the docker is running")

        try:
            container_obj = client.containers.get(container_name)
        except docker.errors.NotFound:
            logging.error("Container {} not found".format(container_name))
            return None
        except docker.errors.APIError:
            logging.error("The API cannot be accessed")
            return None

        if container_obj is None:
            return None

        logs_content = container_obj.logs(stdout=True, stderr=True, stream=False, follow=False)
        if logs_content is None:
            return None

        logs_content = sys_utils.decode_our_err_result(logs_content)
        return logs_content
