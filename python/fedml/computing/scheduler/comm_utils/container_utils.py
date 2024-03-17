import logging
import traceback

import docker
from docker import errors

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
import time


class ContainerUtils(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return ContainerUtils()

    def get_docker_client(self):
        try:
            client = docker.from_env()
        except Exception:
            logging.error("Failed to connect to the docker daemon, please ensure that you have "
                          "installed Docker Desktop or Docker Engine, and the docker is running")
            return None

        return client

    def get_docker_object(self, container_name):
        client = self.get_docker_client()
        if client is None:
            return None

        try:
            container_obj = client.containers.get(container_name)
        except docker.errors.NotFound:
            logging.error("Container {} not found".format(container_name))
            return None
        except docker.errors.APIError:
            logging.error("The API cannot be accessed")
            return None

        return container_obj

    def get_container_logs(self, container_name):
        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return None

        logs_content = container_obj.logs(stdout=True, stderr=True, stream=False, follow=False)
        if logs_content is None:
            return None

        logs_content = sys_utils.decode_our_err_result(logs_content)
        return logs_content

    def get_container_logs_since(self, container_name, since_time: int):
        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return None

        logs_content = container_obj.logs(stdout=True, stderr=True, stream=False, follow=False, since=since_time)
        if logs_content is None:
            return None

        logs_content = sys_utils.decode_our_err_result(logs_content)
        return logs_content

    def remove_container(self, container_name):
        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return False

        client = self.get_docker_client()
        if client is None:
            return False

        try:
            client.api.remove_container(container_obj.id, v=True, force=True)
        except Exception as e:
            return False

        return True

    def restart_container(self, container_name, container_port=2345):
        client = self.get_docker_client()
        if client is None:
            raise Exception("Failed to get docker client.")

        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return False, 0

        try:
            client.api.restart(container=container_obj.id)
            inference_port = self.get_host_port(container_obj, container_port)
            container_obj.reload()
            return container_obj.status == "running", inference_port
        except Exception as e:
            logging.error("Failed to restart container ")

        return False, 0

    def stop_container(self, container_name):
        client = self.get_docker_client()
        if client is None:
            raise Exception("Failed to get docker client.")

        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return False

        try:
            client.api.stop(container=container_obj.id)
            container_obj.reload()
            return container_obj.status != "running"
        except Exception as e:
            logging.error(f"Failed to restart container {traceback.format_exc()}")

        return False

    def start_container(self, container_name, container_port=2345):
        client = self.get_docker_client()
        if client is None:
            raise Exception("Failed to get docker client.")

        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return False, 0

        try:
            if container_obj.status != "running":
                client.api.start(container=container_obj.id)
            inference_port = self.get_host_port(container_obj, container_port)
            container_obj.reload()
            return container_obj.status == "running", inference_port
        except Exception as e:
            logging.error(f"Failed to restart container {traceback.format_exc()}")

        return False, 0

    def get_host_port(self, container_object, container_port, usr_indicated_worker_port=None):
        client = self.get_docker_client()
        if client is None:
            raise Exception("Failed to get docker client.")

        # Get the port allocation
        cnt = 0
        while True:
            cnt += 1
            try:
                if usr_indicated_worker_port is not None:
                    inference_http_port = usr_indicated_worker_port
                    break
                else:
                    # Find the random port
                    port_info = client.api.port(container_object.id, container_port)
                    inference_http_port = port_info[0]["HostPort"]
                    logging.info("inference_http_port: {}".format(inference_http_port))
                    break
            except:
                if cnt >= 5:
                    raise Exception("Failed to get the port allocation")
                time.sleep(3)

        return inference_http_port
    
    @staticmethod
    def get_container_rank_same_model(prefix: str):
        """
        Rank (from 0) for the container that run the same model, i.e.
        running_model_name = hash("model_endpoint_id_{}_name_{}_model_id_{}_name_{}_ver_{}")
        """
        try:
            client = docker.from_env()
        except Exception:
            logging.error("Failed to connect to the docker daemon, please ensure that you have "
                        "installed Docker Desktop or Docker Engine, and the docker is running")
            return -1

        try:
            # ignore_removed need to be set to True, in this case, it will not check deleted containers
            # Which, in high concurrency, will cause API error due to the movement of the containers
            container_list = client.containers.list(all=True, ignore_removed=True)
        except docker.errors.APIError:
            logging.error("The API cannot be accessed")
            return -1

        same_model_container_rank = 0
        for container in container_list:
            if container.name.startswith(prefix):
                same_model_container_rank += 1

        return same_model_container_rank

    def pull_image_with_policy(self, image_pull_policy, image_name, client=None):
        docker_client = self.get_docker_client() if client is None else client
        if docker_client is None:
            raise Exception("Failed to get docker client.")

        if image_pull_policy is None:
            logging.warning("You did not specify the image pull policy, will use the default policy:"
                            "IMAGE_PULL_POLICY_IF_NOT_PRESENT")
            image_pull_policy = SchedulerConstants.IMAGE_PULL_POLICY_IF_NOT_PRESENT

        logging.info(f"Pulling policy is {image_pull_policy}")

        if image_pull_policy == SchedulerConstants.IMAGE_PULL_POLICY_ALWAYS:
            logging.info(f"Pulling the image {image_name}...")
            docker_client.images.pull(image_name)
            logging.info(f"Image {image_name} successfully pulled")
        elif image_pull_policy == SchedulerConstants.IMAGE_PULL_POLICY_IF_NOT_PRESENT:
            try:
                docker_client.images.get(image_name)
            except docker.errors.ImageNotFound:
                logging.info("Image not found, start pulling the image...")
                docker_client.images.pull(image_name)
        else:
            raise Exception(f"Unsupported image pull policy: {image_pull_policy}")

