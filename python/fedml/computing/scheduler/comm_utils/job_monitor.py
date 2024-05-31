import json
import logging
import os
import time
import traceback
import datetime

import re
from dateutil.parser import isoparse

from urllib.parse import urlparse

import asyncio

import fedml
import requests

from fedml import mlops
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.scheduler_core.compute_cache_manager import ComputeCacheManager
from fedml.computing.scheduler.slave import client_data_interface
from fedml.computing.scheduler.master import server_data_interface
from fedml.computing.scheduler.model_scheduler import device_client_data_interface
from fedml.computing.scheduler.model_scheduler import device_server_data_interface
from fedml.core.common.singleton import Singleton

from .container_utils import ContainerUtils
from .job_utils import JobRunnerUtils
from ..model_scheduler.device_http_proxy_inference_protocol import FedMLHttpProxyInference
from ..model_scheduler.device_model_cache import FedMLModelCache
from ..model_scheduler.autoscaler.autoscaler import Autoscaler
from ..model_scheduler.autoscaler.policies import ConcurrentQueryPolicy
from ..model_scheduler.device_model_db import FedMLModelDatabase
from ..model_scheduler.device_mqtt_inference_protocol import FedMLMqttInference
from ..slave import client_constants
from ..master import server_constants
from ..model_scheduler import device_client_constants
from ..model_scheduler import device_server_constants
from fedml.computing.scheduler.model_scheduler.device_http_inference_protocol import FedMLHttpInference
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.core.mlops.mlops_utils import MLOpsLoggingUtils
from fedml.core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ..scheduler_core.endpoint_sync_protocol import FedMLEndpointSyncProtocol

from ..model_scheduler.device_server_constants import ServerConstants


class JobMonitor(Singleton):
    ENDPOINT_CONTAINER_LOG_PREFIX = "endpoint"
    TIME_INTERVAL_FOR_INFERENCE_ON_GATEWAY = 60 * 10
    ENDPOINT_CONTAINER_LOG_SUBDIR = "monitor_endpoint_logs"

    def __init__(self):
        if not hasattr(self, "endpoint_unavailable_counter"):
            self.endpoint_unavailable_counter = dict()
        if not hasattr(self, "released_endpoints"):
            self.released_endpoints = dict()
        if not hasattr(self, "released_runs"):
            self.released_runs = dict()
        if not hasattr(self, "reported_runs"):
            self.reported_runs = dict()
        if not hasattr(self, "reported_runs_on_edges"):
            self.reported_runs_on_edges = dict()
        if not hasattr(self, "mqtt_config"):
            self.mqtt_config = dict()
        if not hasattr(self, "is_first_inference_on_gateway"):
            self.is_first_inference_on_gateway = True
        if not hasattr(self, "last_time_on_inference_gateway"):
            self.last_time_on_inference_gateway = None
        if not hasattr(self, "replica_log_channels"):
            """
                {$job_id: {$edge_id: {$replica_no: $channel_info}}}
                channel_info: {"docker_last_time_stamp": int}
            """
            self.replica_log_channels = dict()

        if not hasattr(self, "endpoints_autoscale_predict_future"):
            self.endpoints_autoscale_predict_future = dict()

    @staticmethod
    def get_instance():
        return JobMonitor()

    def autoscaler_reconcile_after_interval(self):
        """
        For each endpoint,
           if a prediction is pending, or during reconciling (scale out or in) -> wait for next round
           else:
            -> send the scale op (if not None) to MLOps platform
            -> query autoscaler module
        """

        # Set the redis params
        fedml_model_cache = FedMLModelCache.get_instance()
        fedml_model_cache.set_redis_params()
        redis_addr, redis_port, redis_password = \
            fedml_model_cache.get_redis_params()

        # Get all endpoints info
        endpoints_settings_list = fedml_model_cache.get_all_endpoints_user_setting()

        # Init the autoscaler
        autoscaler = Autoscaler(redis_addr, redis_port, redis_password)

        for endpoint_settings in endpoints_settings_list:
            endpoint_state = endpoint_settings["state"]
            if endpoint_state == "DEPLOYED" and endpoint_settings["enable_auto_scaling"]:
                try:    # Should not let one endpoint affect the others
                    logging.info(f"After interval, check the autoscaler for async future list."
                                 f"{self.endpoints_autoscale_predict_future}")
                    # TODO(fedml-dimitris): The policy can be set dynamically or be user specific.
                    # Set the policy, here we use latency, but other metrics are possible as well, such as qps.
                    # For more advanced use cases look for the testing scripts under the autoscaler/test directory.
                    autoscaling_policy_config = \
                        {
                            "current_replicas": int(endpoint_settings["replica_num"]),
                            "min_replicas": int(endpoint_settings["scale_min"]),
                            "max_replicas": int(endpoint_settings["scale_max"]),
                            "queries_per_replica": int(endpoint_settings["target_queries_per_replica"]),
                            "window_size_secs": int(endpoint_settings["aggregation_window_size_seconds"]),
                            "scaledown_delay_secs": int(endpoint_settings["scale_down_delay_seconds"]),
                        }
                    autoscaling_policy = ConcurrentQueryPolicy(**autoscaling_policy_config)

                    e_id, e_name, model_name = endpoint_settings["endpoint_id"], endpoint_settings["endpoint_name"], \
                                                  endpoint_settings["model_name"]

                    logging.info(f"Querying the autoscaler for endpoint {e_id} with user settings {endpoint_settings}.")

                    # For every endpoint we just update the policy configuration.
                    autoscaling_policy.min_replicas = endpoint_settings["scale_min"]
                    autoscaling_policy.max_replicas = endpoint_settings["scale_max"]
                    # We retrieve a list of replicas for every endpoint. The number
                    # of running replicas is the length of that list.
                    current_replicas = len(fedml_model_cache.get_endpoint_replicas_results(e_id))
                    autoscaling_policy.current_replicas = current_replicas
                    logging.info(f"Endpoint {e_id} autoscaling policy: {autoscaling_policy}.")

                    scale_op = autoscaler.scale_operation_endpoint(
                        autoscaling_policy,
                        str(e_id))

                    new_replicas = current_replicas + scale_op.value

                    logging.info(f"Scaling operation {scale_op.value} for endpoint {e_id} .")
                    logging.info(f"New Replicas {new_replicas} for endpoint {e_id} .")
                    logging.info(f"Current Replicas {current_replicas} for endpoint {e_id} .")
                    if current_replicas == new_replicas:
                        # Basically the autoscaler decided that no scaling operation should take place.
                        logging.info(f"No scaling operation for endpoint {e_id}.")
                        return

                    # Should scale in / out
                    curr_version = fedml.get_env_version()

                    if curr_version == "release":
                        mlops_prefix = "https://open.fedml.ai/"
                    elif curr_version == "test":
                        mlops_prefix = "https://open-test.fedml.ai/"
                    else:
                        logging.error(f"Do not support the version {curr_version}.")
                        return
                    autoscale_url_path = "fedmlModelServer/api/v1/endpoint/auto-scale"
                    url = f"{mlops_prefix}{autoscale_url_path}"

                    # Get cached token for authorization of autoscale request
                    cached_token = fedml_model_cache.get_end_point_token(e_id, e_name, model_name)
                    if cached_token is None:
                        logging.error(f"Failed to get the cached token for endpoint {e_id}.")
                        return

                    req_header = {
                        "Authorization": f"Bearer {cached_token}"
                    }
                    req_body = {
                        "endpointId": int(e_id),
                        "replicasDesired": int(new_replicas)
                    }

                    try:
                        logging.info(f"Sending the autoscale request to MLOps platform. url {url}, "
                                     f"body {req_body}., header {req_header}")
                        response = requests.post(
                            url,
                            headers=req_header,
                            json=req_body
                        )
                        if response.status_code != 200:
                            logging.error(f"Failed to send the autoscale request to MLOps platform.")
                        else:
                            logging.info(f"Successfully sent the autoscale request to MLOps platform.")
                    except Exception as e:
                        logging.error(f"Failed to send the autoscale request to MLOps platform. {e}")
                except Exception as e:
                    logging.error(f"Error in autoscaler reconcile after interval. {e}")
                    pass
        return

    @staticmethod
    def monitor_replicas_number():
        """
        [If Master] Report the replica number.
        """
        FedMLModelCache.get_instance().set_redis_params()
        res_frm_db = FedMLModelCache.get_instance().get_all_deployment_result_list()
        res_to_mlops = {}  # endpoint_id -> num_replica

        for endpoint_detail in res_frm_db:
            endpoint_replicas_details = {}
            if isinstance(endpoint_detail, str):
                endpoint_replicas_details = json.loads(endpoint_detail)
                if isinstance(endpoint_replicas_details, str):
                    endpoint_replicas_details = json.loads(endpoint_replicas_details)

            if "result" in endpoint_replicas_details:
                endpoint_replica_details = {}
                if isinstance(endpoint_replicas_details["result"], str):
                    endpoint_replica_details = json.loads(endpoint_replicas_details["result"])

                res_to_mlops[endpoint_replica_details["end_point_id"]] = res_to_mlops.get(
                    endpoint_replica_details["end_point_id"], 0) + 1

        for endpoint_id, num_replica in res_to_mlops.items():
            curr_version = fedml.get_env_version()
            num_replica_url_path = "fedmlModelServer/api/v1/endpoint/replica-info"
            mlops_prefix = fedml._get_backend_service()
            url = f"{mlops_prefix}/{num_replica_url_path}"

            cached_token = FedMLModelCache.get_instance().get_end_point_token_with_eid(endpoint_id)
            if cached_token is None:
                logging.error(f"Failed to get the cached token for endpoint {endpoint_id}.")
                return

            req_header = {
                "Authorization": f"Bearer {cached_token}"
            }
            req_body = {
                "endpointId": int(endpoint_id),
                "replicaNumber": int(num_replica),
                "timestamp": int(time.time() * 1000)
            }

            try:
                response = requests.post(
                    url,
                    headers=req_header,
                    json=req_body
                )
                if response.status_code != 200:
                    logging.error(f"Failed to send the replica number request to MLOps platform.")
                else:
                    logging.debug(f"Successfully sent the replica number request to MLOps platform.")
            except Exception as e:
                logging.error(f"Failed to send the replica number request to MLOps platform. {e}")
        return

    @staticmethod
    def monitor_replicas_perf(edge_id, mqtt_mgr=None):
        """
        For each worker, report the performance of each replica.
        """
        topic_name = "fl_client/mlops/system_performance"

        try:
            fedml_args = mlops.get_fedml_args()
            ComputeCacheManager.get_instance().set_redis_params()
            try:
                device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
            except Exception as e:
                pass
            job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()

            for job in job_list.job_list:
                if job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                        job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
                    MLOpsRuntimeLogDaemon.get_instance(fedml_args).stop_log_processor(job.job_id, int(job.edge_id))
                    continue

                endpoint_json = json.loads(job.running_json) if job.running_json is not None else {}
                model_config = endpoint_json.get("model_config", {})
                model_name = model_config.get("model_name", None)
                model_id = model_config.get("model_id", None)
                model_version = model_config.get("model_version", None)
                endpoint_name = endpoint_json.get("end_point_name", None)

                # Get endpoint container name
                endpoint_container_name_prefix = device_client_constants.ClientConstants.get_endpoint_container_name(
                    endpoint_name, model_name, model_version, job.job_id, model_id, edge_id=job.edge_id
                )

                num_containers = ContainerUtils.get_instance().get_container_rank_same_model(
                    endpoint_container_name_prefix)

                for i in range(num_containers):
                    endpoint_container_name = endpoint_container_name_prefix + f"__{i}"
                    container_perf = ContainerUtils.get_instance().get_container_perf(endpoint_container_name)

                    if container_perf is None:
                        continue

                    metrics_of_all_gpus = []
                    for gpu_id, gpu_perf in container_perf.gpus_stat.items():
                        # gpu_id:int -> {"gpu_utilization", "gpu_memory_allocated", "gpu_temp"}
                        gpu_info = {
                            "gpu_id": gpu_id,
                            "gpu_utilization": round(gpu_perf["gpu_utilization"], 4),
                            "gpu_memory_allocated": round(gpu_perf["gpu_memory_allocated"], 4),
                            "gpu_temp": round(gpu_perf["gpu_temp"], 4),
                            # "gpu_power_usage": round(gpu_perf["gpu_power_usage"], 4),
                            # "gpu_time_spent_accessing_memory": round(gpu_perf["gpu_time_spent_accessing_memory"], 4),
                        }
                        metrics_of_all_gpus.append(gpu_info)

                    # TODO(Raphael): Change to replica-level reporting, now is device-level reporting.
                    device_info_json = {
                        "run_id": job.job_id,
                        "edge_id": job.edge_id,
                        "cpu_utilization": round(container_perf.cpu_percent, 4),
                        "process_memory_in_use": round(container_perf.mem_used_megabytes, 4),
                        "process_memory_available": round(container_perf.mem_avail_megabytes, 4),
                        "process_memory_in_use_size": round(
                            (container_perf.mem_used_megabytes / container_perf.mem_avail_megabytes), 4),
                        "disk_utilization": round(container_perf.blk_write_megabytes, 4),
                        "network_traffic": round(container_perf.network_sent_megabytes, 4),
                        "metrics_of_all_gpus": metrics_of_all_gpus,
                    }

                    message_json = json.dumps(device_info_json)

                    if mqtt_mgr is not None:
                        mqtt_mgr.send_message_json(topic_name, message_json)
                    break   # TODO(Raphael, Asce): Change to replica-level reporting.

        except Exception as e:
            logging.error(f"Exception when monitoring replicas performance on the worker agent. error: {e}")
            pass

    def monitor_slave_run_process_status(self):
        try:
            count = 0
            job_list = client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                # Calc the timeout
                started_time = JobMonitor.get_started_time(job)
                timeout = time.time() - started_time

                job_type = JobRunnerUtils.parse_job_type(job.running_json)
                if job_type is not None and job_type == SchedulerConstants.JOB_TASK_TYPE_DEPLOY:
                    continue

                # Check if all processes of the specific run are exited
                # FIXME: Proactively release the gpu ids when the run processes have not even started yet as the docker
                #  image is being pulled
                # run_process_list = client_constants.ClientConstants.get_learning_process_list(job.job_id)
                # all_run_processes_exited = True if len(run_process_list) <= 0 else False
                # if all_run_processes_exited:
                #     if not self.released_runs.get(str(job.job_id), False):
                #         self.released_runs[str(job.job_id)] = True
                #         # Release the gpu ids
                #         print(
                #             f"[run/device][{job.job_id}/{job.edge_id}] Release gpu resource when run processes has exited on monioring slave runs periodically.")
                #         JobRunnerUtils.get_instance().release_gpu_ids(job.job_id, job.edge_id)

                # Get the timeout threshold
                timeout_threshold = None
                if job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_PROVISIONING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_QUEUED:
                    timeout_threshold = SchedulerConstants.TRAIN_PROVISIONING_TIMEOUT
                elif job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_RUN_STATUS_STARTING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING:
                    timeout_threshold = SchedulerConstants.TRAIN_STARTING_TIMEOUT
                elif job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_RUN_STATUS_RUNNING:
                    timeout_threshold = SchedulerConstants.TRAIN_RUNNING_TIMEOUT
                elif job.status == client_constants.ClientConstants.MSG_MLOPS_RUN_STATUS_STOPPING:
                    timeout_threshold = SchedulerConstants.TRAIN_STOPPING_TIMEOUT

                # If the run processes have exited but run status is not completed and
                # timeout is out of the range, then release gpu ids and report failed status to the master agent.
                # if all_run_processes_exited and not SchedulerConstants.is_run_completed(job.status) and \
                #         timeout_threshold is not None and timeout > timeout_threshold:
                if timeout_threshold is not None and timeout > timeout_threshold:
                    # Report failed status to the master agent
                    mlops.log_training_failed_status(
                        run_id=job.job_id, edge_id=job.edge_id, enable_broadcast=True)

                    print(f"[Slave][{job.job_id}:{job.edge_id}] Due to timeout, release gpu ids and "
                          f"set run status of slave to failed.")

        except Exception as e:
            print(f"Exception when monitoring run process on the slave agent.{traceback.format_exc()}")
            pass

        # Deploy Job Type
        try:
            count = 0
            try:
                device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
            except Exception as e:
                pass
            job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                if job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                        job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED or \
                        job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE or \
                        job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE:
                    if not self.released_endpoints.get(str(job.job_id), False):
                        self.released_endpoints[str(job.job_id)] = True

                        # Release the gpu ids (In case it is not handled by the deployment process)
                        logging.info(
                            f"[endpoint/device][{job.job_id}/{job.edge_id}] "
                            f"Release gpu resource when monitoring worker endpoint periodically.")
                        JobRunnerUtils.get_instance().release_gpu_ids(job.job_id, job.edge_id)

                        # Set the status to MSG_MLOPS_CLIENT_STATUS_ARCHIVED
                        device_client_data_interface.FedMLClientDataInterface.get_instance().save_job_status(
                            job_id=job.job_id,
                            edge_id=job.edge_id,
                            status=device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_ARCHIVED,
                            msg=f"Archive after release the gpu resource. Previous status is {job.status}."
                        )

        except Exception as e:
            logging.error(f"Exception when monitoring endpoint process on the slave agent.{traceback.format_exc()}")
            pass

    @staticmethod
    def get_started_time(job):
        started_time = int(float(job.started_time))
        if started_time <= 0:
            started_time = int(float(job.updated_time))
            if started_time <= 0:
                started_time = time.time()
        return started_time

    def monitor_master_run_process_status(self, server_id, device_info_reporter=None):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            count = 0
            job_list = server_data_interface.FedMLServerDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                # Calc the timeout
                started_time = JobMonitor.get_started_time(job)
                timeout = time.time() - started_time

                # Get the timeout threshold
                timeout_threshold = None
                if job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_PROVISIONING:
                    timeout_threshold = SchedulerConstants.TRAIN_PROVISIONING_TIMEOUT
                elif job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING or \
                        job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING:
                    timeout_threshold = SchedulerConstants.TRAIN_STARTING_TIMEOUT
                elif job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING:
                    timeout_threshold = SchedulerConstants.TRAIN_RUNNING_TIMEOUT
                elif job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_STOPPING:
                    timeout_threshold = SchedulerConstants.TRAIN_STOPPING_TIMEOUT

                # Check if the timout is greater than the threshold value
                if timeout_threshold is not None and timeout > timeout_threshold:
                    if not self.reported_runs.get(str(job.job_id), False):
                        self.reported_runs[str(job.job_id)] = True

                        # Report failed status to the master agent
                        mlops.log_aggregation_failed_status(run_id=job.job_id, edge_id=server_id)
                        print(
                            f"[Master][{job.job_id}:{job.edge_id}:{server_id}] Due to timeout, set run status to failed.")

                # Request all running process list from the edge device.
                running_timeout = timeout_threshold if timeout_threshold is not None else \
                    SchedulerConstants.TRAIN_INIT_TIMEOUT
                if not SchedulerConstants.is_run_completed(job.status) and \
                        timeout > running_timeout:
                    run_completed_on_all_edges = True
                    if job.running_json is None:
                        continue
                    job_run_json = json.loads(job.running_json)
                    edge_ids = job_run_json.get("edgeids", [])
                    for edge_id in edge_ids:
                        device_info = device_info_reporter.get_device_info(job.job_id, edge_id, server_id)
                        if device_info is None:
                            continue
                        run_process_list_map = device_info.get("run_process_list_map", {})
                        run_process_list = run_process_list_map.get(str(job.job_id), [])
                        if len(run_process_list) <= 0:
                            # Report failed status to the master agent
                            if self.reported_runs_on_edges.get(str(job.job_id)) is None:
                                self.reported_runs_on_edges[str(job.job_id)] = dict()
                            if not self.reported_runs_on_edges[str(job.job_id)].get(str(edge_id), False):
                                self.reported_runs_on_edges[str(job.job_id)][str(edge_id)] = True

                                mlops.log_training_failed_status(run_id=job.job_id, edge_id=edge_id)
                                mlops.log_run_log_lines(
                                    job.job_id, edge_id, ["ERROR: Client process exited------------------------------"],
                                    SchedulerConstants.JOB_TASK_TYPE_TRAIN)
                                print(f"[Master][{job.job_id}:{edge_id}] Due to job terminated on the slave agent, "
                                      f"set run status of slave to failed.")
                        else:
                            run_completed_on_all_edges = False

                    # If run completed on all edges, then report run status to the master agent.
                    if run_completed_on_all_edges:
                        # Report failed status to the master agent
                        if not self.reported_runs.get(str(job.job_id), False):
                            self.reported_runs[str(job.job_id)] = True

                            mlops.log_run_log_lines(
                                job.job_id, job.edge_id, ["ERROR: Run failed ------------------------------"],
                                SchedulerConstants.JOB_TASK_TYPE_TRAIN)
                            mlops.log_aggregation_failed_status(run_id=job.job_id, edge_id=server_id)
                            print(f"[Master][{job.job_id}:{job.edge_id}:{server_id}] "
                                  f"Due to job failed on all slave agents, set run status to failed.")

        except Exception as e:
            print(f"Exception when monitoring run process on the master agent.{traceback.format_exc()}")
            pass

    def monitor_slave_endpoint_status(self):
        endpoint_sync_protocol = None
        try:
            count = 0
            try:
                device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
            except Exception as e:
                pass
            FedMLModelDatabase.get_instance().set_database_base_dir(
                device_client_constants.ClientConstants.get_database_dir())
            job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
            agent_config = dict()
            agent_config["mqtt_config"] = self.mqtt_config
            endpoint_sync_protocol = FedMLEndpointSyncProtocol(agent_config=agent_config)
            endpoint_sync_protocol.setup_client_mqtt_mgr()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                try:
                    if job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
                        if not self.released_endpoints.get(str(job.job_id), False):
                            self.released_endpoints[str(job.job_id)] = True

                            # Release the gpu ids
                            print(
                                f"[endpoint/device][{job.job_id}/{job.edge_id}] Release gpu resource when worker "
                                f"endpoint failed on monitoring periodically.")
                            JobRunnerUtils.get_instance().release_gpu_ids(job.job_id, job.edge_id)

                    elif job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
                        endpoint_json = json.loads(job.running_json)
                        model_config = endpoint_json.get("model_config", {})
                        model_name = model_config.get("model_name", None)
                        model_version = model_config.get("model_version", None)
                        model_id = model_config.get("model_id", None)
                        endpoint_name = endpoint_json.get("end_point_name", None)
                        device_ids = endpoint_json.get("device_ids", [])

                        if model_name is None:
                            continue

                        # Get replicas deployment result inside this device
                        deployment_result_list = FedMLModelDatabase.get_instance().get_deployment_result_with_device_id(
                            job.job_id, endpoint_name, model_name, job.edge_id)

                        if deployment_result_list is None:
                            continue

                        # Check the container (replica) ready probe
                        # TODO: Parallel this check
                        rank = -1

                        for deployment_result in deployment_result_list:
                            rank += 1
                            is_endpoint_ready = self._check_and_reset_endpoint_status(
                                job.job_id, job.edge_id, deployment_result, only_check_inference_ready_status=True)

                            # [Hotfix] Under high-concurrency situation, the ready endpoint might not be available
                            # But the container is in health state
                            # In this case, we need to have an exact 503 code, instead of timeout to decide to restart
                            # TODO(Raphael): Split the /ready endpoint and predict endpoint traffic
                            if not self._lenient_check_replica_ready(deployment_result):
                                is_endpoint_ready = False
                            else:
                                is_endpoint_ready = True

                            # Get endpoint container name prefix, prepare for restart
                            endpoint_container_name_prefix = \
                                (device_client_constants.ClientConstants.get_endpoint_container_name(
                                    endpoint_name, model_name, model_version, job.job_id, model_id,
                                    edge_id=job.edge_id))

                            endpoint_container_name = endpoint_container_name_prefix + f"__{rank}"
                            inference_port = -1

                            if is_endpoint_ready:
                                # Though it is ready, we still need to get the port
                                started, inference_port = ContainerUtils.get_instance().start_container(
                                    endpoint_container_name)
                            else:
                                logging.info(
                                    f"======================================"
                                    f"Check endpoint status failed for endpoint {job.job_id} "
                                    f"on device:{job.edge_id} with replica_no:{rank+1}."
                                    f"======================================")
                                # Restart the container if the endpoint is not ready
                                # send unavailable status to the master agent
                                # TODO: Check the callback from the master agent
                                endpoint_sync_protocol.send_sync_inference_info(
                                    device_ids[0], job.edge_id, job.job_id, endpoint_name, model_name,
                                    model_id, model_version, inference_port=None,
                                    disable=True, replica_no=rank+1)

                                # [Critical]
                                # 1. After restart, the "running" status of container does NOT mean the endpoint is
                                # ready. Could be the container is still starting, or the endpoint is not ready.
                                # 2. if local db has status updating, do not restart again
                                result_json = deployment_result
                                status = result_json.get("model_status", None)

                                if (status != device_server_constants.ServerConstants.
                                        MSG_MODELOPS_DEPLOYMENT_STATUS_UPDATING):
                                    # First time restart
                                    started, inference_port = ContainerUtils.get_instance().restart_container(
                                        endpoint_container_name)

                                    # Change the local port for next ready check, avoid restart again
                                    deployment_result["model_status"] = (device_server_constants.ServerConstants.
                                                                         MSG_MODELOPS_DEPLOYMENT_STATUS_UPDATING)
                                    endpoint_sync_protocol.set_local_deployment_status_result(
                                        job.job_id, endpoint_name, model_name, model_version, job.edge_id,
                                        inference_port, None, deployment_result, replica_no=rank+1)

                            # Report the status to the master agent
                            if is_endpoint_ready:
                                assert inference_port != -1
                                deployment_result[
                                    "model_status"] = (device_server_constants.ServerConstants.
                                                       MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)

                                # Send the inference info to the master agent
                                # TODO: Consistency control
                                endpoint_sync_protocol.send_sync_inference_info(
                                    device_ids[0], job.edge_id, job.job_id, endpoint_name, model_name,
                                    model_id, model_version, inference_port, replica_no=rank+1)

                                # Change the local port for next ready check
                                endpoint_sync_protocol.set_local_deployment_status_result(
                                    job.job_id, endpoint_name, model_name, model_version, job.edge_id,
                                    inference_port, None, deployment_result, replica_no=rank+1)

                    elif job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE:
                        # TODO: Bring the offline status online
                        endpoint_json = json.loads(job.running_json)
                        model_config = endpoint_json.get("model_config", {})
                        model_name = model_config.get("model_name", None)
                        model_version = model_config.get("model_version", None)
                        model_id = model_config.get("model_id", None)
                        endpoint_name = endpoint_json.get("end_point_name", None)

                        # Get replicas deployment result inside this device
                        deployment_result = FedMLModelDatabase.get_instance().get_deployment_result_with_device_id(
                            job.job_id, endpoint_name, model_name, job.edge_id)
                        if deployment_result is None:
                            continue

                        status_result = FedMLModelDatabase.get_instance().get_deployment_status_with_device_id(
                            job.job_id, endpoint_name, model_name, job.edge_id)

                        is_endpoint_ready = self._check_and_reset_endpoint_status(
                            job.job_id, job.edge_id, deployment_result, only_check_inference_ready_status=True)
                        if is_endpoint_ready:
                            # Get endpoint container name prefix
                            endpoint_container_name_prefix = device_client_constants.ClientConstants.get_endpoint_container_name(
                                endpoint_name, model_name, model_version, job.job_id, model_id, edge_id=job.edge_id)

                            # Could be multiple containers for the same endpoint
                            num_containers = ContainerUtils.get_instance().get_container_rank_same_model(
                                endpoint_container_name_prefix)

                            for i in range(num_containers):
                                endpoint_container_name = endpoint_container_name_prefix + f"__{i}"
                                started, inference_port = ContainerUtils.get_instance().start_container(
                                    endpoint_container_name)
                                if started and inference_port != 0:
                                    endpoint_sync_protocol.send_sync_inference_info(
                                        device_ids[0], job.edge_id, job.job_id, endpoint_name, model_name,
                                        model_id, model_version, inference_port)

                                    endpoint_sync_protocol.set_local_deployment_status_result(
                                        job.job_id, endpoint_name, model_name, model_version, job.edge_id,
                                        inference_port, status_result, deployment_result)
                    elif job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_RUNNING:
                        endpoint_json = json.loads(job.running_json)
                        model_config = endpoint_json.get("model_config", {})
                        model_name = model_config.get("model_name", None)
                        model_version = model_config.get("model_version", None)
                        model_id = model_config.get("model_id", None)
                        endpoint_name = endpoint_json.get("end_point_name", None)
                        device_ids = endpoint_json.get("device_ids", [])

                        started_time = JobMonitor.get_started_time(job)
                        timeout = time.time() - started_time
                        if timeout > SchedulerConstants.ENDPOINT_DEPLOYMENT_DEPLOYING_TIMEOUT:
                            print(f"[Worker][{job.job_id}:{job.edge_id}] Due to timeout, "
                                  f"set worker status to status "
                                  f"{device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED}.")

                            mlops.log_training_status(
                                device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                                run_id=job.job_id,
                                edge_id=job.edge_id, is_from_model=True, enable_broadcast=True
                            )

                            if not self.released_endpoints.get(str(job.job_id), False):
                                self.released_endpoints[str(job.job_id)] = True

                                # Release the gpu ids
                                print(
                                    f"[endpoint/device][{job.job_id}/{job.edge_id}] Release gpu resource when the worker endpoint runs timeout on monioring periodically.")
                                JobRunnerUtils.get_instance().release_gpu_ids(job.job_id, job.edge_id)

                                # Get endpoint container name prefix
                                endpoint_container_name_prefix = device_client_constants.ClientConstants.get_endpoint_container_name(
                                    endpoint_name, model_name, model_version, job.job_id, model_id, edge_id=job.edge_id)

                                # Could be multiple containers for the same endpoint
                                num_containers = ContainerUtils.get_instance().get_container_rank_same_model(
                                    endpoint_container_name_prefix)

                                for i in range(num_containers):
                                    endpoint_container_name = endpoint_container_name_prefix + f"__{i}"
                                    stopped = ContainerUtils.get_instance().stop_container(endpoint_container_name)

                                print(f"[Worker][{job.job_id}:{job.edge_id}] Release gpu ids.")
                except Exception as e:
                    print(
                        f"[Worker][{job.job_id}:{job.edge_id}] Exception when syncing endpoint process on the slave agent. {traceback.format_exc()}")
        except Exception as e:
            print(f"[Worker] Exception when syncing endpoint process on the slave agent. {traceback.format_exc()}")
            pass
        finally:
            if endpoint_sync_protocol is not None:
                try:
                    endpoint_sync_protocol.release_client_mqtt_mgr()
                except Exception as e:
                    pass

    def _lenient_check_replica_ready(
            self, deployment_result
    ):
        """
        Double-check the replica's liveness using /ready api:
            if 200 -> return True
            [Critical] if timeout -> Could be under high pressure -> return True
            if HTTP_202_ACCEPTED -> unhealthy -> return False
        """
        result_json = deployment_result
        inference_url = result_json.get("model_url", None)

        # Make a curl get to inference_url with timeout 5s
        # TODO(Raphael): Also support PROXY and MQTT to check the readiness
        response_ok = asyncio.run(FedMLHttpInference.is_inference_ready(inference_url, timeout=5))
        if response_ok is None:
            # This means the server return 202
            return False

        # 200 or Timeout
        return True

    def _check_and_reset_endpoint_status(
            self, endpoint_id, device_id, deployment_result, only_check_inference_ready_status=False,
            should_release_gpu_ids=False
    ):
        result_json = deployment_result
        inference_url = result_json.get("model_url", None)
        model_metadata = result_json.get("model_metadata", {})
        input_list = model_metadata.get("inputs", None)
        output_list = []

        # Run inference request to check if endpoint is running normally.
        while True:
            if only_check_inference_ready_status:
                response_ok = self.is_inference_ready(
                    inference_url, timeout=SchedulerConstants.ENDPOINT_INFERENCE_READY_TIMEOUT,
                    device_id=device_id, endpoint_id=endpoint_id
                )
            else:
                response_ok, inference_response = self.inference(
                    device_id, endpoint_id, inference_url, input_list, output_list,
                    timeout=SchedulerConstants.ENDPOINT_STATUS_CHECK_TIMEOUT
                )

            if self.endpoint_unavailable_counter.get(str(endpoint_id)) is None:
                self.endpoint_unavailable_counter[str(endpoint_id)] = 0

            if not response_ok:
                self.endpoint_unavailable_counter[str(endpoint_id)] += 1
            else:
                self.endpoint_unavailable_counter[str(endpoint_id)] = 0
                return True

            # If the endpoint unavailable counter is greater than the threshold value,
            # then try restarting the endpoint and release the gpu ids.
            # If should_release_gpu_ids is True, then release the gpu ids.
            if self.endpoint_unavailable_counter.get(str(endpoint_id), 0) > \
                    SchedulerConstants.ENDPOINT_FAIL_THRESHOLD_VALUE:
                if not self.released_endpoints.get(str(endpoint_id), False) and should_release_gpu_ids:
                    self.released_endpoints[str(endpoint_id)] = True
                    # Release the gpu ids
                    print(
                        f"[endpoint/device][{endpoint_id}/{device_id}] Release gpu resource "
                        f"when the worker endpoint is not ready on monitoring periodically.")
                    JobRunnerUtils.get_instance().release_gpu_ids(endpoint_id, device_id)

                return False
            time.sleep(2)

    def is_inference_ready(self, inference_url, timeout=None, device_id=None, endpoint_id=None, use_mqtt=False):
        response_ok = asyncio.run(FedMLHttpInference.is_inference_ready(inference_url, timeout=timeout))
        if response_ok:
            return response_ok

        if response_ok is None:
            # Internal server can respond, but reply is not ready
            return False

        # Cannot reach the server, will try other protocols
        print(f"Use http health check failed at {inference_url} for device {device_id} and endpoint {endpoint_id}.")

        response_ok = asyncio.run(FedMLHttpProxyInference.is_inference_ready(
            inference_url, timeout=timeout))
        if response_ok:
            print("Use http proxy health check.")
            return response_ok
        print("Use http proxy health check failed.")

        if not use_mqtt:
            return False

        agent_config = dict()
        agent_config["mqtt_config"] = dict()
        agent_config["mqtt_config"]["BROKER_HOST"] = self.mqtt_config["BROKER_HOST"]
        agent_config["mqtt_config"]["BROKER_PORT"] = self.mqtt_config["BROKER_PORT"]
        agent_config["mqtt_config"]["MQTT_USER"] = self.mqtt_config["MQTT_USER"]
        agent_config["mqtt_config"]["MQTT_PWD"] = self.mqtt_config["MQTT_PWD"]
        agent_config["mqtt_config"]["MQTT_KEEPALIVE"] = self.mqtt_config["MQTT_KEEPALIVE"]
        mqtt_inference = FedMLMqttInference(agent_config=agent_config, run_id=endpoint_id)
        response_ok = mqtt_inference.run_mqtt_health_check_with_request(
            device_id, endpoint_id, inference_url, timeout=timeout)

        print(f"Use mqtt health check. return {response_ok}")
        return response_ok

    def inference(
            self, device_id, endpoint_id, inference_url, input_list, output_list,
            inference_type="default", timeout=None):
        try:
            response_ok = asyncio.run(FedMLHttpInference.is_inference_ready(inference_url))
            if response_ok:
                response_ok, inference_response = asyncio.run(FedMLHttpInference.run_http_inference_with_curl_request(
                    inference_url, input_list, output_list, inference_type=inference_type, timeout=timeout))
                print(f"Use http inference. return {response_ok}.")
                return response_ok, inference_response

            response_ok = asyncio.run(FedMLHttpProxyInference.is_inference_ready(inference_url))
            if response_ok:
                response_ok, inference_response = asyncio.run(
                    FedMLHttpProxyInference.run_http_proxy_inference_with_request(
                        endpoint_id, inference_url, input_list, output_list, inference_type=inference_type,
                        timeout=timeout))
                print(f"Use http proxy inference. return {response_ok}.")
                return response_ok, inference_response

            agent_config = dict()
            agent_config["mqtt_config"] = dict()
            agent_config["mqtt_config"]["BROKER_HOST"] = self.mqtt_config["BROKER_HOST"]
            agent_config["mqtt_config"]["BROKER_PORT"] = self.mqtt_config["BROKER_PORT"]
            agent_config["mqtt_config"]["MQTT_USER"] = self.mqtt_config["MQTT_USER"]
            agent_config["mqtt_config"]["MQTT_PWD"] = self.mqtt_config["MQTT_PWD"]
            agent_config["mqtt_config"]["MQTT_KEEPALIVE"] = self.mqtt_config["MQTT_KEEPALIVE"]
            mqtt_inference = FedMLMqttInference(agent_config=agent_config, run_id=endpoint_id)
            response_ok = mqtt_inference.run_mqtt_health_check_with_request(
                device_id, endpoint_id, inference_url, timeout=timeout)
            if response_ok:
                response_ok, inference_response = mqtt_inference.run_mqtt_inference_with_request(
                    device_id, endpoint_id, inference_url, input_list, output_list, inference_type=inference_type,
                    timeout=timeout)

            if not response_ok:
                inference_response = {"error": True,
                                      "message": "Failed to use http, http-proxy and mqtt for inference."}

            print(f"Use mqtt inference. return {response_ok}.")
            return response_ok, inference_response
        except Exception as e:
            inference_response = {"error": True,
                                  "message": f"Exception when using http, http-proxy and mqtt for inference: {traceback.format_exc()}."}
            print("Inference Exception: {}".format(traceback.format_exc()))
            return False, inference_response

        return False, None

    def _check_all_slave_endpoint_status(self, endpoint_id, endpoint_name, model_name,
                                         server_internal_port=ServerConstants.MODEL_INFERENCE_DEFAULT_PORT):
        # Get model deployment result
        is_endpoint_offline = True
        gateway_device_id = None
        gateway_result_payload_for_ready = None
        result_list = FedMLModelCache.get_instance().get_deployment_result_list(
            endpoint_id, endpoint_name, model_name)
        for result_item in result_list:
            result_device_id, _, result_payload = FedMLModelCache.get_instance().get_result_item_info(
                result_item)

            # Check if the endpoint is activated
            endpoint_activated = FedMLModelCache.get_instance().get_end_point_activation(endpoint_id)
            if endpoint_activated:
                # Check if the endpoint is running
                model_url = result_payload.get("model_url", "")
                url_parsed = urlparse(model_url)
                if url_parsed.path.startswith("/inference"):
                    gateway_device_id = result_device_id
                    gateway_result_payload_for_ready = result_payload
                    url_parsed = urlparse(result_payload.get("model_url", ""))
                    gateway_result_payload_for_ready[
                        "model_url"] = f"http://localhost:{server_internal_port}{url_parsed.path}"
                else:
                    if self._check_and_reset_endpoint_status(
                            endpoint_id, result_device_id, result_payload, only_check_inference_ready_status=True,
                            should_release_gpu_ids=False):
                        is_endpoint_offline = False

        if gateway_result_payload_for_ready is not None and is_endpoint_offline is False:
            if not self._check_and_reset_endpoint_status(
                    endpoint_id, gateway_device_id, gateway_result_payload_for_ready,
                    only_check_inference_ready_status=True, should_release_gpu_ids=False):
                is_endpoint_offline = True

        return not is_endpoint_offline

    def monitor_master_endpoint_status(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            count = 0
            FedMLModelCache.get_instance().set_redis_params()
            try:
                device_server_data_interface.FedMLServerDataInterface.get_instance().create_job_table()
            except Exception as e:
                pass
            job_list = device_server_data_interface.FedMLServerDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                endpoint_status = FedMLModelCache.get_instance().get_end_point_status(job.job_id)
                endpoint_json = json.loads(job.running_json) if job.running_json is not None else {}
                model_config = endpoint_json.get("model_config", {})
                model_name = model_config.get("model_name", None)
                endpoint_name = endpoint_json.get("end_point_name", None)

                if endpoint_status == device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                    # Release the gpu ids
                    print(
                        f"[endpoint/device][{job.job_id}/{job.edge_id}] Release gpu resource when the master endpoint failed on monitoring periodically.")
                    JobRunnerUtils.get_instance().release_gpu_ids(job.job_id, job.edge_id)
                elif endpoint_status == device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
                    if model_name is None:
                        continue
                    try:
                        # If the endpoint is offline, then report offline status to the MLOps.
                        model_config_parameters = endpoint_json.get("parameters", {})
                        server_internal_port = model_config_parameters.get("server_internal_port",
                                                                           ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
                        is_endpoint_online = self._check_all_slave_endpoint_status(job.job_id, endpoint_name,
                                                                                   model_name, server_internal_port)
                        if not is_endpoint_online:
                            print(f"[Master][{job.job_id}] Due to all worker is offline, set endpoint status to "
                                  f"offline after deployed .")
                            mlops.log_endpoint_status(
                                job.job_id,
                                device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
                            FedMLModelCache.get_instance().set_end_point_status(
                                job.job_id, endpoint_name,
                                device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)

                    except Exception as e:
                        print(f"[Master][{job.job_id}] Exception when check endpoint status: "
                              f"{traceback.format_exc()}")
                elif endpoint_status == device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE:
                    # If the endpoint is offline, then report offline status to the MLOps.
                    model_config_parameters = model_config.get("parameters", {})
                    server_internal_port = model_config_parameters.get("server_internal_port",
                                                                       ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
                    is_endpoint_online = self._check_all_slave_endpoint_status(
                        job.job_id, endpoint_name, model_name, server_internal_port)
                    if is_endpoint_online:
                        print(f"[Master][{job.job_id}] Due to all worker is from offline to online, "
                              f"set endpoint status to online.")
                        mlops.log_endpoint_status(
                            job.job_id,
                            device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)
                        FedMLModelCache.get_instance().set_end_point_status(
                            job.job_id, endpoint_name,
                            device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)

        except Exception as e:
            print(f"Exception when syncing endpoint process on the master agent {traceback.format_exc()}.")
            pass

    def monitor_endpoint_logs(self):
        try:
            fedml_args = mlops.get_fedml_args()
            ComputeCacheManager.get_instance().set_redis_params()
            count = 0
            try:
                device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
            except Exception as e:
                pass
            number_of_finished_jobs = 0
            job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                # [Deprecated] log_virtual_edge_id = int(job.edge_id) * 2
                if job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                        job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
                    MLOpsRuntimeLogDaemon.get_instance(fedml_args).stop_log_processor(job.job_id, int(job.edge_id))
                    continue

                if job.status != device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
                    continue
                number_of_finished_jobs += 1
                if number_of_finished_jobs >= 100:
                    break

                endpoint_json = json.loads(job.running_json) if job.running_json is not None else {}
                model_config = endpoint_json.get("model_config", {})
                model_name = model_config.get("model_name", None)
                model_id = model_config.get("model_id", None)
                model_version = model_config.get("model_version", None)
                endpoint_name = endpoint_json.get("end_point_name", None)

                log_file_dir = os.path.join(
                    device_client_constants.ClientConstants.get_log_file_dir(),
                    JobMonitor.ENDPOINT_CONTAINER_LOG_SUBDIR)
                log_file_path, program_prefix = MLOpsLoggingUtils.build_log_file_path_with_run_params(
                    job.job_id, int(job.edge_id), log_file_dir, is_server=False,
                    log_file_prefix=JobMonitor.ENDPOINT_CONTAINER_LOG_PREFIX,
                )

                # Get endpoint container name
                endpoint_container_name_prefix = device_client_constants.ClientConstants.get_endpoint_container_name(
                    endpoint_name, model_name, model_version, job.job_id, model_id, edge_id=job.edge_id
                )

                num_containers = ContainerUtils.get_instance().get_container_rank_same_model(
                    endpoint_container_name_prefix)

                is_job_container_running = False
                for i in range(num_containers):
                    endpoint_container_name = endpoint_container_name_prefix + f"__{i}"

                    if job.job_id in self.replica_log_channels and \
                            job.edge_id in self.replica_log_channels[job.job_id] and \
                            i in self.replica_log_channels[job.job_id][job.edge_id]:
                        # Get log since last time stamp
                        channel_info = self.replica_log_channels[job.job_id][job.edge_id][i]
                        if channel_info.get("docker_last_time_stamp") is not None:
                            endpoint_logs = ContainerUtils.get_instance().get_container_logs_since(
                                endpoint_container_name, since_time=channel_info.get("docker_last_time_stamp"),
                                timestamps=True
                            )
                            if endpoint_logs is not None:
                                channel_info["docker_last_time_stamp"] = int(time.time())
                        else:
                            endpoint_logs = ContainerUtils.get_instance().get_container_logs(endpoint_container_name,
                                                                                             timestamps=True)
                            if endpoint_logs is not None:
                                channel_info["docker_last_time_stamp"] = int(time.time())
                    else:
                        # First time to get the logs
                        self.replica_log_channels[job.job_id] = self.replica_log_channels.get(job.job_id, dict())
                        self.replica_log_channels[job.job_id][job.edge_id] = self.replica_log_channels[job.job_id].get(
                            job.edge_id, dict())
                        self.replica_log_channels[job.job_id][job.edge_id][i] = dict()
                        self.replica_log_channels[job.job_id][job.edge_id][i]["docker_last_time_stamp"] = (
                            int(time.time()))
                        endpoint_logs = ContainerUtils.get_instance().get_container_logs(endpoint_container_name,
                                                                                         timestamps=True)

                        # Sync Time by setting the offset
                        if endpoint_logs is not None:
                            t_sec_offset = ContainerUtils.get_instance().get_container_deploy_time_offset(
                                endpoint_container_name)
                            self.replica_log_channels[job.job_id][job.edge_id][i]["deploy_container_t_offset"] = (
                                t_sec_offset)

                    if (endpoint_logs is None or endpoint_logs == "\n" or endpoint_logs == "\r\n" or
                            endpoint_logs == "\r" or endpoint_logs == "" or endpoint_logs == " "):
                        continue

                    is_job_container_running = True

                    # Append containers log to the same log file (as they are in the same job & device)
                    with open(log_file_path, "a") as f:
                        line_of_logs = endpoint_logs.split("\n")

                        # Add NTP offset
                        channel_info = self.replica_log_channels[job.job_id][job.edge_id][i]
                        t_sec_offset = channel_info.get("deploy_container_t_offset", None)

                        for line in line_of_logs:
                            if line == "":
                                continue

                            try:
                                container_time = line.split(" ")[0]
                                nano_second_str = container_time.split(".")[1][:9]
                                t_datetime_obj = isoparse(container_time)

                                # ISSUE: this will cause the timestamp is not correct.
                                #if t_sec_offset is not None:
                                #    t_datetime_obj = t_datetime_obj + datetime.timedelta(seconds=t_sec_offset)
                            except Exception as e:
                                logging.error(f"Exception when parsing the container log time {e}")
                                t_datetime_obj = datetime.datetime.now()
                                nano_second_str = "000000000"

                            t_sec = t_datetime_obj.strftime("%a, %d %b %Y %H:%M:%S")
                            t_nano_sec = f"[{t_sec}.{nano_second_str}]"

                            device_replica_prefix = f"[FedML-Replica @device-id-{job.edge_id} @replica-rank-{i}]"
                            log_level_prefix = "[INFO]"    # Use default level to represent the docker log level
                            program_location_prefix = "[Container Logs]"  # Use default location

                            prefix = (f"{device_replica_prefix} {t_nano_sec} "
                                      f"{log_level_prefix} {program_location_prefix} ")

                            # replace prefix
                            line = line.replace(f"{container_time} ", prefix)
                            f.write(line + "\n")

                if is_job_container_running and not MLOpsRuntimeLogDaemon.get_instance(fedml_args). \
                        is_log_processor_running(job.job_id, int(job.edge_id)):
                    setattr(fedml_args, "log_file_dir", os.path.dirname(log_file_path))
                    setattr(fedml_args, "log_file_path", log_file_path)
                    setattr(fedml_args, "run_id", job.job_id)
                    setattr(fedml_args, "edge_id", int(job.edge_id))

                    MLOpsRuntimeLogDaemon.get_instance(fedml_args).log_file_dir = os.path.dirname(log_file_path)

                    # Init (Write to) the config file
                    MLOpsRuntimeLog(fedml_args).init_logs()

                    # Init log processor if not running
                    MLOpsRuntimeLogDaemon.get_instance(fedml_args).start_log_processor(
                        job.job_id, int(job.edge_id),
                        log_source=device_client_constants.ClientConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT,
                        log_file_prefix=JobMonitor.ENDPOINT_CONTAINER_LOG_PREFIX
                    )

        except Exception as e:
            print(f"Exception when syncing endpoint log to MLOps {traceback.format_exc()}.")


