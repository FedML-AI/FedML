
import json
import logging
import os
import shutil
import time
import traceback
import urllib
import yaml
from abc import ABC
from dataclasses import dataclass

from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
from fedml.computing.scheduler.comm_utils.network_util import return_this_device_connectivity_type

from fedml.core.mlops import MLOpsRuntimeLog
from fedml.computing.scheduler.comm_utils import file_utils
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants

from .device_client_constants import ClientConstants
from .device_model_cache import FedMLModelCache
from ..scheduler_core.general_constants import GeneralConstants
from ..slave.base_slave_job_runner import FedMLBaseSlaveJobRunner
from .device_model_deployment import start_deployment
from .device_model_db import FedMLModelDatabase
from .device_replica_handler import FedMLDeviceReplicaHandler


@dataclass
class DeployJobConfig:
    def __init__(self, request_json, edge_id):
        self._ori_req_json = request_json
        self.edge_id = edge_id
        self.run_id = request_json["end_point_id"]
        self.end_point_name = request_json["end_point_name"]
        self.device_ids = request_json["device_ids"]
        self.master_id = request_json["device_ids"][0]
        self.master_ip = request_json["master_node_ip"]
        self.model_config = request_json["model_config"]
        self.model_name = self.model_config["model_name"]
        self.model_id = self.model_config["model_id"]
        self.model_version = self.model_config["model_version"]
        self.model_config_parameters = request_json["parameters"]
        self.inference_port = self.model_config_parameters.get("worker_internal_port",
                                                               ClientConstants.MODEL_INFERENCE_DEFAULT_PORT)
        self.inference_port_external = self.model_config_parameters.get("worker_external_port", self.inference_port)
        self.inference_engine = self.model_config_parameters.get("inference_engine",
                                                                 ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEFAULT)
        self.version_diff_dict = request_json.get("replica_version_diff", {}).get(str(self.edge_id), {})
        self.encrypted_api_key = request_json.get(ServerConstants.USER_ENCRYPTED_API_KEY, "")
        self.gpu_per_replica = int(request_json.get("gpu_per_replica", 1))

        # Dynamic config
        self.model_storage_local_path = ""  # After downloading the model package
        self.worker_ip = ""                 # Dynamic ip of the worker
        self.replica_rank = None


@dataclass
class DeployJobResult:
    def __init__(self, deployment_result=ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                 inference_output_url="", model_metadata=None, model_config=None, connectivity_type=None):
        self.deployment_result = deployment_result
        self.inference_output_url = inference_output_url
        self.model_metadata = model_metadata
        self.model_config = model_config
        self.connectivity_type = connectivity_type


class FedMLDeployWorkerJobRunner(FedMLBaseSlaveJobRunner, ABC):

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0,
                 cuda_visible_gpu_ids_str=None):
        FedMLBaseSlaveJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=ClientConstants.get_data_dir(),
            agent_package_download_dir=ClientConstants.get_model_package_dir(),
            agent_package_unzip_dir=GeneralConstants.get_package_unzip_dir(ClientConstants.get_package_download_dir()),
            agent_log_file_dir=ClientConstants.get_log_file_dir()
        )

        self.is_deployment_runner = True
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
        urllib.request.urlretrieve(package_url, local_package_file,
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
        end_point_name = self.request_json["end_point_name"]
        model_version = model_config["model_version"]

        # Generate the model package dir for downloading.
        model_version = model_version.replace(" ", "-")     # Avoid using space for folder name
        model_version = model_version.replace(":", "-")     # Since docker mount will conflict with ":"
        local_package_path = ClientConstants.get_model_package_dir()
        os.makedirs(local_package_path, exist_ok=True)
        this_run_model_dir = f"{run_id}_{end_point_name}_{model_name}_{model_version}"
        this_run_model_full_path = os.path.join(local_package_path, this_run_model_dir)
        self.agent_package_download_dir = this_run_model_full_path
        self.agent_package_unzip_dir = this_run_model_full_path

        # Retrieve model package or model binary file.
        if self.model_is_from_open:
            unzip_package_path, model_bin_file = self.retrieve_binary_model_file(model_name, model_storage_url)
        else:
            unzip_package_path = self.retrieve_and_unzip_package(model_name, model_storage_url)
            model_bin_file = FedMLDeployWorkerJobRunner.get_model_bin_file(unzip_package_path)

        # Load the config to memory
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
        # Parse json
        job_conf = DeployJobConfig(self.request_json, self.edge_id)
        self.run_id = job_conf.run_id

        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        logging.info(f"[Worker] Received model deployment request from master for endpoint {job_conf.run_id}.")
        self.replica_handler = FedMLDeviceReplicaHandler(self.edge_id, self.request_json)
        if self.replica_handler is not None:
            logging.info("\n================= Worker replica Handler ======================\n"
                         f"Reconcile with num diff {self.replica_handler.replica_num_diff}\n"
                         f"and version diff {self.replica_handler.replica_version_diff}\n"
                         "===============================================================\n")
        else:
            logging.error(f"[Worker] Replica handler is None.")
            return False

        self.check_runner_stop_event()

        # Report the deployment status to mlops
        self.status_reporter.report_client_id_status(
            self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING,
            is_from_model=True, running_json=json.dumps(self.request_json), run_id=job_conf.run_id)
        self.status_reporter.report_client_id_status(
            self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_RUNNING,
            is_from_model=True, run_id=job_conf.run_id)

        self.check_runner_stop_event()

        # Reconcile the replica number (op: add, remove)
        prev_rank, op, op_num = self.replica_handler.reconcile_num_replica()

        # Reconcile the replica version (op: update)
        replica_rank_to_update = []
        if not op:
            replica_rank_to_update, op = self.replica_handler.reconcile_replica_version()

        if not op:
            logging.info("[Worker] No need to reconcile.")
            return True
        logging.info("\n================ Worker Reconcile Operations ======================\n"
                     f"                op: {op}; op num: {op_num}.\n"
                     "===================================================================\n")

        model_version = job_conf.model_version
        if op == "rollback":
            logging.info("Try to find backup package to rollback...")
            version_rollback_to = None
            for replica_no, rollback_ops in job_conf.version_diff_dict.items():
                version_rollback_to = rollback_ops["new_version"]     # Note that new_version is the version to rollback
                break
            if version_rollback_to is None:
                logging.error(f"No old version found for run_id: {self.run_id} "
                              f"edge_id: {self.edge_id}, rollback failed. No old version found in request_json.")
                return False
            model_version = version_rollback_to

        # Construct the parent folder name for the package
        model_version_formatted = model_version.replace(" ", "-")
        model_version_formatted = model_version_formatted.replace(":", "-")
        models_root_dir = ClientConstants.get_model_package_dir()
        parent_fd = f"{job_conf.run_id}_{job_conf.end_point_name}_{job_conf.model_name}_{model_version_formatted}"

        # Check if the package is already downloaded
        unzip_package_path = ""
        if os.path.exists(os.path.join(models_root_dir, parent_fd)):
            unzip_package_path = self.find_previous_downloaded_pkg(os.path.join(models_root_dir, parent_fd))

        # Download the package if not found
        if unzip_package_path == "":
            logging.info("Download and unzip model to local...")
            unzip_package_path, _, _ = \
                self.update_local_fedml_config(job_conf.run_id, job_conf.model_config, job_conf.model_config_parameters)
            if unzip_package_path is None:
                logging.info("Failed to update local fedml config.")
                self.check_runner_stop_event()
                self.status_reporter.report_client_id_status(
                    self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                    is_from_model=True, run_id=job_conf.run_id)
                return False

        if not os.path.exists(unzip_package_path):
            logging.error("Failed to unzip file.")
            self.check_runner_stop_event()
            self.status_reporter.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                is_from_model=True, run_id=job_conf.run_id)
            return False

        job_conf.model_storage_local_path = unzip_package_path

        self.check_runner_stop_event()

        running_model_name, inference_output_url, inference_model_version, model_metadata, model_config = \
            "", "", model_version, {}, {}

        # ip and connectivity
        job_conf.worker_ip = GeneralConstants.get_ip_address(self.request_json)
        connectivity = return_this_device_connectivity_type()

        deploy_res = DeployJobResult(connectivity_type=connectivity)
        if op == "add":
            for rank in range(prev_rank + 1, prev_rank + 1 + op_num):
                job_conf.replica_rank = rank

                try:
                    start_deployment(job_conf, deploy_res)
                except Exception as e:
                    inference_output_url = ""
                    logging.error(f"[Worker] Exception at deployment: {traceback.format_exc()}")

                if inference_output_url == "":
                    logging.error("[Worker] Failed to deploy the model.")
                    _ = self.send_deployment_results(job_conf, deploy_res)
                    self.status_reporter.run_id = self.run_id
                    raise Exception("[Worker] Failed to deploy the model.")
                else:
                    logging.info("Finished deployment, continue to send results to master...")
                    result_payload = self.send_deployment_results(job_conf, deploy_res)

                    FedMLModelDatabase.get_instance().set_deployment_result(job_conf, result_payload)

                    logging.info(f"Deploy replica {rank + 1} / {prev_rank + 1 + op_num} successfully.")

            self.status_reporter.run_id = self.run_id
            self.status_reporter.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                is_from_model=True, run_id=self.run_id)
            return True
        elif op == "remove":
            for rank_to_delete in range(prev_rank, prev_rank - op_num, -1):
                self.replica_handler.remove_replica(rank_to_delete)

                FedMLModelCache.get_instance().set_redis_params()
                replica_occupied_gpu_ids_str = FedMLModelCache.get_instance().get_replica_gpu_ids(
                    run_id, end_point_name, model_name, self.edge_id, rank_to_delete + 1)

                replica_occupied_gpu_ids = json.loads(replica_occupied_gpu_ids_str)

                JobRunnerUtils.get_instance().release_partial_job_gpu(run_id, self.edge_id, replica_occupied_gpu_ids)

                FedMLModelDatabase.get_instance().delete_deployment_result_with_device_id_and_rank(
                    run_id, end_point_name, model_name, self.edge_id, rank_to_delete)

                # Report the deletion msg to master
                _ = self.send_deployment_results(
                    end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DELETED,
                    model_id, model_name, inference_output_url, model_version, inference_port_external,
                    inference_engine, model_metadata, model_config, replica_no=rank_to_delete + 1)

                time.sleep(1)
                self.status_reporter.run_id = self.run_id
                self.status_reporter.report_client_id_status(
                    self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                    is_from_model=True, run_id=self.run_id)

                # TODO: If delete all replica, then delete the job and related resources
                if rank_to_delete == 0:
                    pass
            return True
        elif op == "update" or op == "rollback":
            # Update is combine of delete and add
            for rank in replica_rank_to_update:
                # Delete a replica (container) if exists
                self.replica_handler.remove_replica(rank)

                FedMLModelCache.get_instance().set_redis_params()
                replica_occupied_gpu_ids_str = FedMLModelCache.get_instance().get_replica_gpu_ids(
                    run_id, end_point_name, model_name, self.edge_id, rank + 1)

                replica_occupied_gpu_ids = json.loads(replica_occupied_gpu_ids_str)
                logging.info(f"Release gpu ids {replica_occupied_gpu_ids} for update / rollback.")

                # TODO (Raphael) check if this will allow another job to seize the gpu during high concurrency:
                try:
                    JobRunnerUtils.get_instance().release_partial_job_gpu(
                        run_id, self.edge_id, replica_occupied_gpu_ids)
                except Exception as e:
                    if op == "rollback":
                        pass
                    else:
                        logging.error(f"Failed to release gpu ids {replica_occupied_gpu_ids} for update.")
                        return False

                # Delete the deployment result from local db
                FedMLModelDatabase.get_instance().delete_deployment_result_with_device_id_and_rank(
                    run_id, end_point_name, model_name, self.edge_id, rank)

                logging.info(f"Delete replica with no {rank + 1} successfully.")
                time.sleep(1)

                # Add a replica (container)
                # TODO: Reduce the duplicated code
                logging.info(f"Start to deploy the model with replica no {rank + 1} ...")
                try:
                    running_model_name, inference_output_url, inference_model_version, model_metadata, model_config = \
                        start_deployment(
                            end_point_id=inference_end_point_id, end_point_name=end_point_name, model_id=model_id,
                            model_version=model_version, model_storage_local_path=unzip_package_path,
                            inference_model_name=model_name, inference_engine=inference_engine,
                            infer_host=worker_ip, master_ip=master_ip, edge_id=self.edge_id,
                            master_device_id=device_ids[0], replica_rank=rank,
                            gpu_per_replica=int(self.replica_handler.gpu_per_replica), request_json=self.request_json
                        )
                except Exception as e:
                    inference_output_url = ""
                    logging.error(f"Exception at deployment: {traceback.format_exc()}")

                if inference_output_url == "":
                    logging.error("Failed to deploy the model...")

                    # Release the gpu occupancy
                    FedMLModelCache.get_instance().set_redis_params()
                    replica_occupied_gpu_ids_str = FedMLModelCache.get_instance().get_replica_gpu_ids(
                        run_id, end_point_name, model_name, self.edge_id, rank + 1)
                    logging.info(f"Release gpu ids {replica_occupied_gpu_ids_str} for "
                                 f"failed deployment of replica no {rank + 1}.")

                    if replica_occupied_gpu_ids_str is not None:
                        replica_occupied_gpu_ids = json.loads(replica_occupied_gpu_ids_str)
                        JobRunnerUtils.get_instance().release_partial_job_gpu(
                            run_id, self.edge_id, replica_occupied_gpu_ids)

                    self.send_deployment_results(
                        end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                        model_id, model_name, inference_output_url, inference_model_version, inference_port,
                        inference_engine, model_metadata, model_config)

                    self.status_reporter.run_id = self.run_id
                    self.status_reporter.report_client_id_status(
                        self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                        is_from_model=True, run_id=self.run_id)
                    return False
                else:
                    logging.info("Finished deployment, continue to send results to master...")
                    result_payload = self.send_deployment_results(
                        end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                        model_id, model_name, inference_output_url, model_version, inference_port_external,
                        inference_engine, model_metadata, model_config, replica_no=rank + 1,
                        connectivity=connectivity
                    )

                    if inference_port_external != inference_port:  # Save internal port to local db
                        logging.info("inference_port_external {} != inference_port {}".format(
                            inference_port_external, inference_port))
                        result_payload = self.construct_deployment_results(
                            end_point_name, self.edge_id, ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                            model_id, model_name, inference_output_url, model_version, inference_port,
                            inference_engine, model_metadata, model_config, replica_no=rank + 1,
                            connectivity=connectivity
                        )

                    FedMLModelDatabase.get_instance().set_deployment_result(
                        run_id, end_point_name, model_name, model_version, self.edge_id,
                        json.dumps(result_payload), replica_no=rank + 1)

                    logging.info(f"Update replica with no {rank + 1}  successfully. Op num {op_num}")
                    time.sleep(5)
            time.sleep(1)
            self.status_reporter.run_id = self.run_id
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
                                     model_metadata, model_config, replica_no=1,
                                     connectivity=ClientConstants.WORKER_CONNECTIVITY_TYPE_DEFAULT):
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
                                      "connectivity_type": connectivity,
                                      }
        return deployment_results_payload

    def send_deployment_results(self, job_conf, deploy_res):
        end_point_name = job_conf.end_point_name
        device_id = job_conf.edge_id
        model_id = job_conf.model_id
        model_name = job_conf.model_name
        model_version = job_conf.model_version
        inference_port = job_conf.inference_port
        inference_engine = job_conf.inference_engine

        model_status = deploy_res.deployment_result
        model_inference_url = deploy_res.inference_output_url
        model_metadata = deploy_res.model_metadata
        model_config = deploy_res.model_config
        connectivity = deploy_res.connectivity_type

        # TODO: think about how to do gang scheduling
        replica_no = job_conf.replica_rank

        deployment_results_topic = "model_device/model_device/return_deployment_result/{}/{}".format(
            self.run_id, device_id)

        deployment_results_payload = self.construct_deployment_results(
            end_point_name, device_id, model_status,
            model_id, model_name, model_inference_url,
            model_version, inference_port, inference_engine,
            model_metadata, model_config, replica_no=replica_no, connectivity=connectivity)

        logging.info("[client] send_deployment_results: topic {}, payload {}.".format(deployment_results_topic,
                                                                                      deployment_results_payload))
        self.message_center.send_message_json(deployment_results_topic, json.dumps(deployment_results_payload))
        return deployment_results_payload

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

    @staticmethod
    def find_previous_downloaded_pkg(parent_dir: str) -> str:
        """
        Find a folder inside parent_dir that contains the fedml_model_config.yaml file.
        """
        res = file_utils.find_file_inside_folder(parent_dir, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        if res is not None:
            # return the parent folder of res
            return os.path.dirname(res)
        else:
            return ""
