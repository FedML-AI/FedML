import logging
import os
import traceback
import datetime
from typing import List

from dateutil.parser import isoparse

import docker
from docker import errors

from fedml.computing.scheduler.comm_utils.hardware_utils import HardwareUtil
from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
import time
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.computing.scheduler.comm_utils.scheduler_utils import SchedulerUtils
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.crictl_utils import CriClient


class ContainerUtils(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return ContainerUtils()

    def get_docker_client(self):
        if SchedulerUtils.is_using_k8s():
            return None
        
        try:
            client = docker.from_env()
        except Exception:
            logging.error("Failed to connect to the docker daemon, please ensure that you have "
                          "installed Docker Desktop or Docker Engine, and the docker is running")
            return None

        return client

    def get_docker_object(self, container_name):
        if SchedulerUtils.is_using_k8s():
            return None
        
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

    def get_container_logs(self, container_name, timestamps=False):
        if SchedulerUtils.is_using_k8s():
            cri_client = CriClient()
            container_id = cri_client.get_container_id()
            if container_id:
                logs = cri_client.get_logs(container_id, since=None, follow=False, timestamps=timestamps)
                if logs:
                    return logs
            return None

        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return None

        logs_content = container_obj.logs(stdout=True, stderr=True, stream=False, follow=False, timestamps=timestamps)
        if logs_content is None:
            return None

        logs_content = sys_utils.decode_our_err_result(logs_content)
        return logs_content

    def get_container_logs_since(self, container_name, since_time: int, timestamps=False):
        if SchedulerUtils.is_using_k8s():
            cri_client = CriClient()
            container_id = cri_client.get_container_id()
            if container_id:
                logs = cri_client.get_logs(container_id, since=since_time, follow=False, timestamps=timestamps)
                if logs:
                    return logs
            return None

        container_obj = self.get_docker_object(container_name)
        if container_obj is None:
            return None

        logs_content = container_obj.logs(stdout=True, stderr=True, stream=False, follow=False, since=since_time,
                                          timestamps=timestamps)
        if logs_content is None:
            return None

        logs_content = sys_utils.decode_our_err_result(logs_content)
        return logs_content

    def remove_container(self, container_name):
        if SchedulerUtils.is_using_k8s():
            # no need to remove container in k8s scheduler
            return False

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

    def restart_container(self, container_name, container_port=ClientConstants.PORT_INSIDE_CONTAINER_DEFAULT):
        if SchedulerUtils.is_using_k8s():
            # no need to restart container in k8s scheduler
            # in one pod: inference_port is equal to the container_port inside the container, use localhost:port to access
            return True, container_port

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
        if SchedulerUtils.is_using_k8s():
            # no need to remove container in k8s scheduler
            return False
        
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

    def start_container(self, container_name, container_port=ClientConstants.PORT_INSIDE_CONTAINER_DEFAULT):
        if SchedulerUtils.is_using_k8s():
            # no need to restart container in k8s scheduler
            # in one pod: inference_port is equal to the container_port inside the container, use localhost:port to access
            return False, container_port
        
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
        if SchedulerUtils.is_using_k8s():
            return None

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
                    break
            except:
                if cnt >= 5:
                    raise Exception("Failed to get the port allocation")
                time.sleep(3)

        return inference_http_port

    @staticmethod
    def get_container_rank_same_model(prefix: str):
        if SchedulerUtils.is_using_k8s():
            # only one deployment (rank 0) in k8s scheduler, so the rank size is 1
            return SchedulerUtils.get_replicate_num_per_pod()

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
        if SchedulerUtils.is_using_k8s():
            return
        
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

    class ContainerMetrics:
        def __init__(self, cpu_percent, mem_used_megabytes, mem_avail_megabytes, network_recv_megabytes,
                     network_sent_megabytes, blk_read_megabytes, blk_write_megabytes, timestamp, gpus_stat):
            self.cpu_percent = cpu_percent
            self.mem_used_megabytes = mem_used_megabytes
            self.mem_avail_megabytes = mem_avail_megabytes
            self.network_recv_megabytes = network_recv_megabytes
            self.network_sent_megabytes = network_sent_megabytes
            self.blk_read_megabytes = blk_read_megabytes
            self.blk_write_megabytes = blk_write_megabytes
            self.timestamp = timestamp
            self.gpus_stat = gpus_stat

        def show(self):
            logging.info(f"CPU: {self.cpu_percent}%")
            logging.info(f"Memory: {self.mem_used_megabytes}GB / {self.mem_avail_megabytes}GB")
            logging.info(f"Network: {self.network_recv_megabytes}MB / {self.network_sent_megabytes}MB")
            logging.info(f"Disk I/O: {self.blk_read_megabytes}MB / {self.blk_write_megabytes}MB")
            logging.info(f"Timestamp: {self.timestamp}")
            logging.info(f"GPU: {self.gpus_stat}")

    def get_container_perf(self, c_name) -> ContainerMetrics:
        """
        CPU Device-Type Related (Use docker stats cmd outside the container):
        CPU %     MEM USAGE / LIMIT     MEM %     NET I/O          BLOCK I/O
        0.26%     8.703GiB / 503.5GiB   1.73%     17.4GB / 176MB   545kB / 20.9GB

        GPU: We currently use HardwareUtil to get the GPU stats on host machine since one GPU is not
        shared by multiple containers
        """
        if SchedulerUtils.is_using_k8s():
            return self.get_container_perf_in_k8s(c_name)

        client = self.get_docker_client()
        container = client.containers.get(c_name)

        # Get stats
        stats = container.stats(stream=False, decode=False)

        # Calculate the memory usage
        mem_bytes_used = stats["memory_stats"]["usage"]
        mem_bytes_avail = stats["memory_stats"]["limit"]
        mem_gb_used = round(mem_bytes_used / (1024 * 1024 * 1024), 1)
        mem_gb_avail = round(mem_bytes_avail / (1024 * 1024 * 1024), 1)

        # Calculate the CPU usage
        cpu_percent = 0.0
        cpu_count = 1
        try:
            cpu_count = len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
        except KeyError:
            try:
                cpu_count = stats["cpu_stats"]["online_cpus"]
            except KeyError:
                try:
                    cpu_count = os.cpu_count()
                except Exception as e:
                    pass

        cpu_delta = (float(stats["cpu_stats"]["cpu_usage"]["total_usage"]) -
                     float(stats["precpu_stats"]["cpu_usage"]["total_usage"]))
        system_delta = (float(stats["cpu_stats"]["system_cpu_usage"]) -
                        float(stats["precpu_stats"]["system_cpu_usage"]))

        if system_delta > 0.0:
            cpu_percent = cpu_delta / system_delta * 100.0 * cpu_count
            cpu_percent = round(cpu_percent, 2)     # Keep only 2 decimal points

        # Calculate the network usage
        recv_bytes, sent_bytes = 0, 0
        try:
            for key, value in stats["networks"].items():
                # key: eth0 is the default network interface
                recv_bytes += value.get("rx_bytes", 0)
                sent_bytes += value.get("tx_bytes", 0)
        except Exception as e:
            logging.error(f"Failed to get network usage: {e}")
            pass

        recv_megabytes, sent_megabytes = round(recv_bytes / (1024 * 1024), 1), round(sent_bytes / (1024 * 1024), 1)

        # Calculate disk I/O usage
        blk_read_bytes, blk_write_bytes = 0, 0
        try:
            for element in stats["blkio_stats"]["io_service_bytes_recursive"]:
                if element["op"] == "Read":
                    blk_read_bytes += element["value"]
                elif element["op"] == "Write":
                    blk_write_bytes += element["value"]
        except Exception as e:
            logging.error(f"Failed to get disk I/O usage: {e}")
            pass

        blk_read_bytes, blk_write_bytes = (
            round(blk_read_bytes / (1024 * 1024), 1), round(blk_write_bytes / (1024 * 1024), 1))

        # Calculate the gpu usage
        gpus_stat = self.generate_container_gpu_stats(container_name=c_name)

        # Record timestamp
        timestamp = stats["read"]

        return ContainerUtils.ContainerMetrics(cpu_percent, mem_gb_used, mem_gb_avail, recv_megabytes, sent_megabytes,
                                               blk_read_bytes, blk_write_bytes, timestamp, gpus_stat)

    def get_container_perf_in_k8s(self, c_name: str) -> ContainerMetrics:
        """
        Get container performance metrics in k8s environment.
        """
        try:
            cri_client = CriClient()
            container_id = cri_client.get_container_id()
            
            if not container_id:
                logging.error("Failed to get container ID")
                return None
            
            stats = cri_client.get_container_stats(container_id)
            if not stats:
                logging.error("No stats data returned from crictl")
                return None
            
            # Get network stats using container PID
            network_recv_mb, network_sent_mb = cri_client.get_network_stats(container_id)
            
            # Get block I/O stats
            blk_read_mb, blk_write_mb = cri_client.get_blkio_stats(container_id)
            
            # Calculate the gpu usage
            gpus_stat = self.generate_container_gpu_stats(c_name)

            return ContainerUtils.ContainerMetrics(
                cpu_percent=stats['cpu_percent'],
                mem_used_megabytes=stats['mem_used_mb'],
                mem_avail_megabytes=stats['mem_avail_mb'],
                network_recv_megabytes=network_recv_mb,
                network_sent_megabytes=network_sent_mb,
                blk_read_megabytes=blk_read_mb,
                blk_write_megabytes=blk_write_mb,
                timestamp=stats['timestamp'],
                gpus_stat=gpus_stat
            )
            
        except Exception as e:
            logging.error(f"Error getting container performance metrics: {str(e)}")
            return None

    def generate_container_gpu_stats(self, container_name):
        client = self.get_docker_client()
        gpu_ids = HardwareUtil.get_docker_gpu_ids_by_container_name(container_name=container_name, docker_client=client)
        gpu_stats = self.gpu_stats(gpu_ids)
        return gpu_stats

    @staticmethod
    def gpu_stats(gpu_ids: List[int]):
        utilz, memory, temp = None, None, None
        gpu_stats_map = {}  # gpu_id: int -> {"gpu_utilization", "gpu_memory_allocated", "gpu_temp"}
        gpu_ids = set(gpu_ids)
        try:
            for gpu in HardwareUtil.get_gpus():
                if gpu.id in gpu_ids:
                    gpu_stats_map[gpu.id] = {
                        "gpu_utilization": gpu.load * 100,
                        "gpu_memory_allocated": gpu.memoryUsed / gpu.memoryTotal * 100,
                        "gpu_temp": gpu.temperature,
                        # "gpu_power_usage": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,   # in watts
                        # "gpu_time_spent_accessing_memory": utilz.memory   # in ms
                    }
        except Exception as e:
            logging.error(f"Failed to get GPU stats: {e}")

        return gpu_stats_map

    @staticmethod
    def get_container_deploy_time_offset(container_name) -> int:
        """
        Diff between the host machine's time and the container's time, in seconds
        """
        if SchedulerUtils.is_using_k8s():
            return 0
        
        time_diff = 0
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
            logs_content = container.logs(stdout=True, stderr=True, stream=False, follow=False, timestamps=True)
            logs_content = sys_utils.decode_our_err_result(logs_content)
            line_of_logs = logs_content.split("\n")

            for line in line_of_logs:
                if line == "":
                    continue

                container_time = line.split(" ")[0]
                nano_second_str = container_time.split(".")[1][:9]
                t_container_datetime_obj = isoparse(container_time)
                curr_host_time = datetime.datetime.now()

                # Calculate the time difference between the container time and the host time
                # The time difference is in seconds
                time_diff = (curr_host_time - t_container_datetime_obj.replace(tzinfo=None)).total_seconds()
                break
        except Exception as e:
            logging.error(f"Failed to get container deploy time offset: {e}")

        return time_diff
