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
    
    @staticmethod
    def get_container_rank_same_model(prefix:str):
        '''
        Rank (from 0) for the container that run the same model, i.e.
        running_model_name = hash(
            "model_endpoint_id_{}_name_{}_model_id_{}_name_{}_ver_{}"
        )
        '''
        try:
            client = docker.from_env()
        except Exception:
            logging.error("Failed to connect to the docker daemon, please ensure that you have "
                        "installed Docker Desktop or Docker Engine, and the docker is running")
            return -1

        try:
            container_list = client.containers.list()
        except docker.errors.APIError:
            logging.error("The API cannot be accessed")
            return -1

        same_model_container_rank = 0
        for container in container_list:
            if container.name.startswith(prefix):
                same_model_container_rank += 1

        return same_model_container_rank

