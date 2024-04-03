
import json
import logging
import os
import shutil
import time
import traceback
import urllib
from abc import ABC
from urllib.parse import urljoin, urlparse
import yaml
from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
from fedml.core.mlops import MLOpsRuntimeLog
from .device_client_constants import ClientConstants
from .device_model_cache import FedMLModelCache
from ..scheduler_core.general_constants import GeneralConstants
from ..slave.base_slave_job_runner import FedMLBaseSlaveJobRunner
from .device_model_deployment import start_deployment
from .device_model_db import FedMLModelDatabase
from .device_replica_handler import FedMLDeviceReplicaHandler


class FedMLDeployWorkerJobRunner(FedMLBaseSlaveJobRunner, ABC):

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0,
                 cuda_visible_gpu_ids_str=None):
        FedMLBaseSlaveJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=ClientConstants.get_data_dir(),
            agent_package_download_dir=ClientConstants.get_package_download_dir(),
            agent_package_unzip_dir=GeneralConstants.get_package_unzip_dir(ClientConstants.get_package_download_dir()),
            agent_log_file_dir=ClientConstants.get_log_file_dir()
        )

        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"
        self.model_is_from_open = False
        self.replica_handler = None

    # Override
    def _generate_job_runner_instance(self, args, run_id=None, request_json=None, agent_config=None, edge_id=None):
        return FedMLDeployWorkerJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=self.agent_config, edge_id=edge_id
        )

    # Override
    def _generate_extend_queue_list(self):
        return None

    def retrieve_binary_model_file(self, package_name, package_url):
        local_package_path = ClientConstants.get_model_package_dir()
        if not os.path.exists(local_package_path):
            os.makedirs(local_package_path, exist_ok=True)
        unzip_package_path = ClientConstants.get_model_dir()
        local_package_file = "{}".format(os.path.join(local_package_path, package_name))
        if os.path.exists(local_package_file):
            os.remove(local_package_file)
        package_url_without_query_path = urljoin(package_url, urlparse(package_url).path)
        urllib.request.urlretrieve(package_url_without_query_path, local_package_file,
                                   reporthook=self.package_download_progress)

        unzip_package_path = os.path.join(unzip_package_path, package_name)
        if not os.path.exists(unzip_package_path):
            os.makedirs(unzip_package_path, exist_ok=True)
        dst_model_file = os.path.join(unzip_package_path, package_name)
        if os.path.exists(local_package_file):
            shutil.copy(local_package_file, dst_model_file)

        return unzip_package_path, dst_model_file

    @staticmethod
    def get_model_bin_file(unzip_package_full_path):
        unzip_package_path = os.path.dirname(unzip_package_full_path)
        model_bin_file = os.path.join(unzip_package_path, "fedml_model.bin")
        return model_bin_file

    def update_local_fedml_config(self, run_id, model_config, model_config_parameters=None):
        model_name = model_config["model_name"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        inference_engine = model_config.get("inference_engine", 0)
        inference_end_point_id = run_id

        # Retrieve model package or model binary file.
        if self.model_is_from_open:
            unzip_package_path, model_bin_file = self.retrieve_binary_model_file(model_name, model_storage_url)
        else:
            unzip_package_path = self.retrieve_and_unzip_package(model_name, model_storage_url)
            model_bin_file = FedMLDeployWorkerJobRunner.get_model_bin_file(unzip_package_path)

        # Load the config to memory
        package_conf_object = {}
        fedml_local_config_file = os.path.join(unzip_package_path, "fedml_model_config.yaml")

        # Inject the config from UI to pkg yaml
        package_conf_object = model_config_parameters

        # Save the config to local
        with open(fedml_local_config_file, "w") as f:
            yaml.dump(package_conf_object, f)

        logging.info("The package_conf_object is {}".format(package_conf_object))

        return unzip_package_path, model_bin_file, package_conf_object

    def download_model_package(self, package_name, package_url):
        # Copy config file from the client
        unzip_package_path = self.retrieve_and_unzip_package(
            package_name, package_url
        )

        return unzip_package_path

    # Override
    def run_impl(self, run_extend_queue_list, sender_message_center,
                 listener_message_queue, status_center_queue):
        run_id = self.request_json["end_point_id"]
        end_point_name = self.request_json["end_point_name"]
        token = self.request_json["token"]
        user_id = self.request_json["user_id"]
        user_name = self.request_json["user_name"]
        device_ids = self.request_json["device_ids"]
        device_objs = self.request_json["device_objs"]
        master_ip = self.request_json["master_node_ip"]

        model_config = self.request_json["model_config"]
        model_name = model_config["model_name"]
        model_id = model_config["model_id"]
        model_version = model_config["model_version"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        model_config_parameters = self.request_json["parameters"]

        self.replica_handler = FedMLDeviceReplicaHandler(self.edge_id, self.request_json)

        inference_port = model_config_parameters.get("worker_internal_port",
                                                     ClientConstants.MODEL_INFERENCE_DEFAULT_PORT)
        inference_port_external = model_config_parameters.get("worker_external_port", inference_port)

        if "using_triton" in model_config_parameters and model_config_parameters["using_triton"]:
            inference_engine = ClientConstants.INFERENCE_ENGINE_TYPE_INT_TRITON
        else:
            inference_engine = ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEFAULT

        logging.info("[Critical] The inference_engine is: {}".format(inference_engine))

        self.model_is_from_open = True if model_config.get("is_from_open", 0) == 1 else False
        if self.model_is_from_open:
            model_net_url = model_config["model_net_url"]
        inference_end_point_id = run_id
        use_gpu = "gpu"  # TODO: Get GPU from device infos
        memory_size = "4096m"  # TODO: Get Memory size for each instance

        self.mlops_metrics.report_sys_perf(self.args, self.agent_config["mqtt_config"], run_id=run_id)

        self.check_runner_stop_event()

        logging.info("model deployment request: {}".format(self.request_json))

        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        self.status_reporter.report_client_id_status(
            self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING,
            is_from_model=True, running_json=json.dumps(self.request_json), run_id=run_id)

        self.status_reporter.report_client_id_status(
            self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_RUNNING,
            is_from_model=True, run_id=run_id)

        self.check_runner_stop_event()

        # update local config with real time parameters from server and dynamically replace variables value
        logging.info("download and unzip model to local...")
        unzip_package_path, model_bin_file, fedml_config_object = \
            self.update_local_fedml_config(run_id, model_config, model_config_parameters)
        if unzip_package_path is None or fedml_config_object is None:
            logging.info("failed to update local fedml config.")
            self.check_runner_stop_event()
            self.status_reporter.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                is_from_model=True, run_id=run_id)
            return False

        logging.info("check downloaded packages...")
        if not os.path.exists(unzip_package_path):
            logging.info("failed to unzip file.")
            self.check_runner_stop_event()
            self.status_reporter.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                is_from_model=True, run_id=run_id)
            return False

        # download model net and load into the torch model
        model_from_open = None
        self.model_is_from_open = None

        logging.info("start the model deployment...")
        self.check_runner_stop_event()
        running_model_name, inference_output_url, inference_model_version, model_metadata, model_config = \
            "", "", model_version, {}, {}

        # Reconcile the replica number (op: add, remove)
        prev_rank, op, op_num = self.replica_handler.reconcile_num_replica()

        # Reconcile the replica version (op: update)
        replica_rank_to_update = []
        if not op:
            replica_rank_to_update, op = self.replica_handler.reconcile_replica_version()

        if not op:
            logging.info("No need to reconcile.")
            return True

        if op == "add":
            worker_ip = GeneralConstants.get_ip_address(self.request_json)
            for rank in range(prev_rank+1, prev_rank+1+op_num):
                # TODO: Support Rollback if this for loop failed
                try:
                    running_model_name, inference_output_url, inference_model_version, model_metadata, model_config = \
                        start_deployment(
                            inference_end_point_id, end_point_name, model_id, model_version,
                            unzip_package_path, model_bin_file, model_name, inference_engine,
                            ClientConstants.INFERENCE_HTTP_PORT,
                            ClientConstants.INFERENCE_GRPC_PORT,
                            ClientConstants.INFERENCE_METRIC_PORT,
                            use_gpu, memory_size,
                            ClientConstants.INFERENCE_CONVERTOR_IMAGE,
                            ClientConstants.INFERENCE_SERVER_IMAGE,
                            worker_ip,
                            self.model_is_from_open, model_config_parameters,
                            model_from_open,
                            token,
                            master_ip, self.edge_id, master_device_id=device_ids[0], replica_rank=rank,
                            gpu_per_replica=int(self.replica_handler.gpu_per_replica)
                        )
                except Exception as e:
                    inference_output_url = ""
                    logging.error(f"Exception at deployment: {traceback.format_exc()}")

                if inference_output_url == "":
                    logging.error("failed to deploy the model...")

                    result_payload = self.send_deployment_results(
                        end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                        model_id, model_name, inference_output_url, inference_model_version, inference_port,
                        inference_engine, model_metadata, model_config)

                    self.status_reporter.report_client_id_status(
                        self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                        is_from_model=True, run_id=self.run_id)
                    return False
                else:
                    logging.info("finished deployment, continue to send results to master...")
                    result_payload = self.send_deployment_results(
                        end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                        model_id, model_name, inference_output_url, model_version, inference_port_external,
                        inference_engine, model_metadata, model_config, replica_no=rank + 1)

                    if inference_port_external != inference_port:  # Save internal port to local db
                        logging.info("inference_port_external {} != inference_port {}".format(
                            inference_port_external, inference_port))
                        result_payload = self.construct_deployment_results(
                            end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                            model_id, model_name, inference_output_url, model_version, inference_port,
                            inference_engine, model_metadata, model_config, replica_no=rank + 1)

                    FedMLModelDatabase.get_instance().set_deployment_result(
                        run_id, end_point_name, model_name, model_version, self.edge_id,
                        json.dumps(result_payload), replica_no=rank + 1)

                    logging.info(f"Deploy replica {rank+1} / {prev_rank+1+op_num} successfully.")
                    time.sleep(5)

            time.sleep(1)
            self.status_reporter.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                is_from_model=True, run_id=self.run_id)
            return True
        elif op == "remove":
            for rank_to_delete in range(prev_rank, prev_rank-op_num, -1):
                self.replica_handler.remove_replica(rank_to_delete)

                FedMLModelCache.get_instance().set_redis_params()
                replica_occupied_gpu_ids_str = FedMLModelCache.get_instance().get_replica_gpu_ids(
                    run_id, end_point_name, model_name, self.edge_id, rank_to_delete+1)

                replica_occupied_gpu_ids = json.loads(replica_occupied_gpu_ids_str)

                JobRunnerUtils.get_instance().release_partial_job_gpu(run_id, self.edge_id, replica_occupied_gpu_ids)

                FedMLModelDatabase.get_instance().delete_deployment_result_with_device_id_and_rank(
                    run_id, end_point_name, model_name, self.edge_id, rank_to_delete)

                # Report the deletion msg to master
                result_payload = self.send_deployment_results(
                    end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DELETED,
                    model_id, model_name, inference_output_url, model_version, inference_port_external,
                    inference_engine, model_metadata, model_config, replica_no=rank_to_delete + 1)

                time.sleep(1)
                self.status_reporter.report_client_id_status(
                    self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                    is_from_model=True, run_id=self.run_id)

                # TODO: If delete all replica, then delete the job and related resources
                if rank_to_delete == 0:
                    pass
            return True
        elif op == "update":
            # Update is combine of delete and add
            worker_ip = GeneralConstants.get_ip_address(self.request_json)
            for rank in replica_rank_to_update:
                # Delete the container
                self.replica_handler.remove_replica(rank)

                FedMLModelCache.get_instance().set_redis_params()
                replica_occupied_gpu_ids_str = FedMLModelCache.get_instance().get_replica_gpu_ids(
                    run_id, end_point_name, model_name, self.edge_id, rank + 1)

                replica_occupied_gpu_ids = json.loads(replica_occupied_gpu_ids_str)

                JobRunnerUtils.get_instance().release_partial_job_gpu(run_id, self.edge_id, replica_occupied_gpu_ids)

                # Delete the deployment result from local db
                FedMLModelDatabase.get_instance().delete_deployment_result_with_device_id_and_rank(
                    run_id, end_point_name, model_name, self.edge_id, rank)

                time.sleep(1)

                # Add the container
                # TODO: Reduce the duplicated code
                try:
                    running_model_name, inference_output_url, inference_model_version, model_metadata, model_config = \
                        start_deployment(
                            inference_end_point_id, end_point_name, model_id, model_version,
                            unzip_package_path, model_bin_file, model_name, inference_engine,
                            ClientConstants.INFERENCE_HTTP_PORT,
                            ClientConstants.INFERENCE_GRPC_PORT,
                            ClientConstants.INFERENCE_METRIC_PORT,
                            use_gpu, memory_size,
                            ClientConstants.INFERENCE_CONVERTOR_IMAGE,
                            ClientConstants.INFERENCE_SERVER_IMAGE,
                            worker_ip,
                            self.model_is_from_open, model_config_parameters,
                            model_from_open,
                            token,
                            master_ip, self.edge_id, master_device_id=device_ids[0], replica_rank=rank,
                            gpu_per_replica=int(self.replica_handler.gpu_per_replica)
                        )
                except Exception as e:
                    inference_output_url = ""
                    logging.error(f"Exception at deployment: {traceback.format_exc()}")

                if inference_output_url == "":
                    logging.error("failed to deploy the model...")

                    result_payload = self.send_deployment_results(
                        end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                        model_id, model_name, inference_output_url, inference_model_version, inference_port,
                        inference_engine, model_metadata, model_config)

                    self.status_reporter.report_client_id_status(
                        self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                        is_from_model=True, run_id=self.run_id)

                    return False
                else:
                    logging.info("finished deployment, continue to send results to master...")
                    result_payload = self.send_deployment_results(
                        end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                        model_id, model_name, inference_output_url, model_version, inference_port_external,
                        inference_engine, model_metadata, model_config, replica_no=rank + 1)

                    if inference_port_external != inference_port:  # Save internal port to local db
                        logging.info("inference_port_external {} != inference_port {}".format(
                            inference_port_external, inference_port))
                        result_payload = self.construct_deployment_results(
                            end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                            model_id, model_name, inference_output_url, model_version, inference_port,
                            inference_engine, model_metadata, model_config, replica_no=rank + 1)

                    FedMLModelDatabase.get_instance().set_deployment_result(
                        run_id, end_point_name, model_name, model_version, self.edge_id,
                        json.dumps(result_payload), replica_no=rank + 1)

                    logging.info(f"Update replica with no {rank + 1}  successfully. Op num {op_num}")
                    time.sleep(5)
            time.sleep(1)
            self.status_reporter.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                is_from_model=True, run_id=self.run_id)
            return True

        else:
            # The delete op will be handled by callback_delete_deployment
            logging.error(f"Unsupported op {op} with op num {op_num}")
            return False

    def construct_deployment_results(self, end_point_name, device_id, model_status,
                                     model_id, model_name, model_inference_url,
                                     model_version, inference_port, inference_engine,
                                     model_metadata, model_config, replica_no=1):
        deployment_results_payload = {"end_point_id": self.run_id, "end_point_name": end_point_name,
                                      "model_id": model_id, "model_name": model_name,
                                      "model_url": model_inference_url, "model_version": model_version,
                                      "port": inference_port,
                                      "inference_engine": inference_engine,
                                      "model_metadata": model_metadata,
                                      "model_config": model_config,
                                      "model_status": model_status,
                                      "inference_port": inference_port,
                                      "replica_no": replica_no,
                                      }
        return deployment_results_payload

    def construct_deployment_status(self, end_point_name, device_id,
                                    model_id, model_name, model_version,
                                    model_inference_url, model_status,
                                    inference_port=ClientConstants.MODEL_INFERENCE_DEFAULT_PORT,
                                    replica_no=1,     # start from 1
                                    ):
        deployment_status_payload = {"end_point_id": self.run_id, "end_point_name": end_point_name,
                                     "device_id": device_id,
                                     "model_id": model_id, "model_name": model_name,
                                     "model_version": model_version,
                                     "model_url": model_inference_url, "model_status": model_status,
                                     "inference_port": inference_port,
                                     "replica_no": replica_no,
                                     }
        return deployment_status_payload

    def send_deployment_results(self, end_point_name, device_id, model_status,
                                model_id, model_name, model_inference_url,
                                model_version, inference_port, inference_engine,
                                model_metadata, model_config, replica_no=1):
        deployment_results_topic = "model_device/model_device/return_deployment_result/{}".format(device_id)
        deployment_results_payload = self.construct_deployment_results(
            end_point_name, device_id, model_status,
            model_id, model_name, model_inference_url,
            model_version, inference_port, inference_engine,
            model_metadata, model_config, replica_no=replica_no)

        logging.info("[client] send_deployment_results: topic {}, payload {}.".format(deployment_results_topic,
                                                               deployment_results_payload))
        self.message_center.send_message_json(deployment_results_topic, json.dumps(deployment_results_payload))
        return deployment_results_payload

    def send_deployment_status(self, end_point_name, device_id,
                               model_id, model_name, model_version,
                               model_inference_url, model_status,
                               inference_port=ClientConstants.MODEL_INFERENCE_DEFAULT_PORT,
                               replica_no=1,     # start from 1
                               ):
        deployment_status_topic = "model_device/model_device/return_deployment_status/{}".format(device_id)
        deployment_status_payload = self.construct_deployment_status(
            end_point_name, device_id,
            model_id, model_name, model_version,
            model_inference_url, model_status,
            inference_port=inference_port,
            replica_no=replica_no)

        logging.info("[client] send_deployment_status: topic {}, payload {}.".format(deployment_status_topic,
                                                                                     deployment_status_payload))
        self.message_center.send_message_json(deployment_status_topic, json.dumps(deployment_status_payload))
        return deployment_status_payload

    def reset_devices_status(self, edge_id, status):
        self.status_reporter.run_id = self.run_id
        self.status_reporter.edge_id = edge_id
        self.status_reporter.report_client_id_status(
            edge_id, status, is_from_model=True, run_id=self.run_id)

    # Override
    def get_download_package_info(self, packages_config=None):
        model_name = packages_config["model_name"]
        model_storage_url = packages_config["model_storage_url"]
        return model_name, model_storage_url

    # Override
    def build_dynamic_args(self, run_id, run_config, package_conf_object, base_dir):
        pass

    # Override
    def build_dynamic_constrain_variables(self, run_id, run_config):
        pass
