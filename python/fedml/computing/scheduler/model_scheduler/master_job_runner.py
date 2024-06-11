import copy
import json
import logging
import os
import time
import queue
import traceback
from abc import ABC
from multiprocessing import Queue

import fedml
from fedml.core.mlops import MLOpsRuntimeLog, MLOpsConfigs
from fedml.core.mlops.mlops_runtime_log import MLOpsFormatter
from .device_client_constants import ClientConstants
from .device_model_cache import FedMLModelCache
from .device_server_constants import ServerConstants
from .device_server_data_interface import FedMLServerDataInterface
from ..comm_utils import sys_utils
from ..comm_utils.run_process_utils import RunProcessUtils
from ..comm_utils.sys_utils import get_python_program
from ..scheduler_core.general_constants import GeneralConstants
from ..master.base_master_job_runner import FedMLBaseMasterJobRunner
from .device_replica_controller import FedMLDeviceReplicaController
from .job_runner_msg_sender import FedMLDeployJobRunnerMsgSender


class FedMLDeployMasterJobRunner(FedMLBaseMasterJobRunner, FedMLDeployJobRunnerMsgSender, ABC):
    default_redis_addr = "local"
    default_redis_port = "6379"
    default_redis_password = "fedml_default"

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0,
                 cuda_visible_gpu_ids_str=None):
        FedMLDeployJobRunnerMsgSender.__init__(self)
        FedMLBaseMasterJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=ServerConstants.get_data_dir(),
            agent_package_download_dir=ServerConstants.get_package_download_dir(),
            agent_package_unzip_dir=GeneralConstants.get_package_unzip_dir(ServerConstants.get_package_download_dir()),
            agent_log_file_dir=ServerConstants.get_log_file_dir()
        )

        self.is_deployment_runner = True
        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"
        self.inference_gateway_process = None
        self.monitor_process = None
        self.replica_controller = None
        self.deployed_replica_payload = None
        self.slave_deployment_results_map = dict()
        self.deployment_result_queue = Queue()
        self.is_fresh_endpoint = True

    # Override
    def _generate_job_runner_instance(self, args, run_id=None, request_json=None, agent_config=None, edge_id=None, ):
        return FedMLDeployMasterJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=self.agent_config, edge_id=edge_id
        )

    # Override
    def _generate_extend_queue_list(self):
        return [self.deployment_result_queue]

    # Override
    def run_impl(
            self, edge_id_status_queue, edge_device_info_queue, run_metrics_queue,
            run_event_queue, run_artifacts_queue, run_logs_queue, edge_device_info_global_queue,
            run_extend_queue_list=None, sender_message_queue=None, listener_message_queue=None,
            status_center_queue=None
    ):
        # Parse the model parameters.
        run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
            model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
            inference_end_point_id, use_gpu, memory_size, model_version, inference_port = \
            FedMLDeployMasterJobRunner.parse_model_run_params(self.request_json)
        self.run_id = run_id
        self.is_fresh_endpoint = self.request_json.get("is_fresh_endpoint", True)

        # Print request parameters.
        logging.info("model deployment request: {}".format(self.request_json))
        logging.info("send deployment stages...")

        # Generate the replica controller object
        self.replica_controller = FedMLDeviceReplicaController(self.edge_id, self.request_json)

        # Start the process to report system performance(cpu,memory,etc.) to MLOps
        # TODO(Raphael): This measurement is for the host machine. Change to container's metrics
        self.mlops_metrics.report_sys_perf(self.args, self.agent_config["mqtt_config"], run_id=run_id)

        # Check if we should stop the runner
        self.check_runner_stop_event()

        # Send stage: MODEL_DEPLOYMENT_STAGE4 = "ForwardRequest2Slave"
        self.send_deployment_stages(
            self.run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE4["index"],
            ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"], ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"],
            message_center=self.message_center)

        # Init the runtime logs
        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        # Report server running status
        logging.info("report deployment status...")
        self.check_runner_stop_event()
        self.status_reporter.report_server_id_status(
            run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING,
            is_from_model=True, running_json=json.dumps(self.request_json),
            server_agent_id=self.edge_id, server_id=self.edge_id, edge_id=self.edge_id)
        self.send_deployment_status(
            self.run_id, end_point_name, model_name, "",
            ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYING,
            message_center=self.message_center)

        # start unified inference gateway process if not started
        FedMLDeployMasterJobRunner.start_device_inference_gateway(inference_port=inference_port)

        # start inference monitor process
        FedMLDeployMasterJobRunner.stop_device_inference_monitor(
            run_id, end_point_name, model_id, model_name, model_version)
        FedMLDeployMasterJobRunner.start_device_inference_monitor(
            run_id, end_point_name, model_id, model_name, model_version)

        # Changed the status to "IDLE"
        self.status_reporter.report_server_id_status(
            run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
            is_from_model=True, server_agent_id=self.edge_id, server_id=self.edge_id, edge_id=self.edge_id)

        # Check if we should stop the runner
        logging.info("send the model inference request to slave devices...")
        self.check_runner_stop_event()

        # Forward deployment request to slave devices
        # Handle "op:add" && "op:remove"
        devices_sent_add_or_remove_msg = self.send_deployment_start_request_to_edges()

        # Handle "op:update"
        try:
            devices_sent_update_remove_msg = self.send_first_scroll_update_msg()

            if len(devices_sent_add_or_remove_msg) == 0 and len(devices_sent_update_remove_msg) == 0:
                # No device is added, updated or removed
                logging.info("No device is added, updated or removed. No action needed for reconciliation.")
                ip = GeneralConstants.get_ip_address(self.request_json)
                master_port = os.getenv("FEDML_MASTER_PORT", None)
                if master_port is not None:
                    inference_port = int(master_port)
                model_inference_port = inference_port
                if ip.startswith("http://") or ip.startswith("https://"):
                    model_inference_url = "{}/api/v1/predict".format(ip)
                else:
                    model_inference_url = "http://{}:{}/api/v1/predict".format(ip, model_inference_port)

                self.send_deployment_status(
                    run_id, end_point_name, model_name, model_inference_url,
                    ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                    message_center=self.message_center
                )

                # Set setting to "DEPLOYED" for autoscaling service reference
                FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
                FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                    update_user_setting_replica_num(end_point_id=run_id, state="DEPLOYED")

                # Complete the job runner
                self.trigger_completed_event()

                return
        except Exception as e:
            logging.error(f"Failed to send first scroll update message due to {e}.")
            logging.error(f"Exception traceback {traceback.format_exc()}.")

        logging.info("Start waiting for result callback from workers ...")

        self.deployment_result_queue = run_extend_queue_list[0]
        while True:
            self.check_runner_stop_event()

            try:
                deployment_result = self.deployment_result_queue.get(block=False, timeout=0.2)
                result_topic = deployment_result.get("topic", None)
                result_payload = deployment_result.get("payload", None)
                self.process_deployment_result_message(topic=result_topic, payload=result_payload)
            except queue.Empty as e:  # If queue is empty, then continue
                pass

            time.sleep(0.5)

    def save_deployment_result(self, topic=None, payload=None):
        self.deployment_result_queue.put({"topic": topic, "payload": payload})

    def process_deployment_result_message(self, topic=None, payload=None):
        # Parse the parameters
        topic_splits = str(topic).split('/')
        device_id = topic_splits[-1]
        payload_json = json.loads(payload)
        end_point_id = payload_json["end_point_id"]
        end_point_name = payload_json["end_point_name"]
        model_id = payload_json["model_id"]
        model_name = payload_json["model_name"]
        model_version = payload_json["model_version"]
        model_status = payload_json["model_status"]
        replica_no = payload_json.get("replica_no", None)  # "no" Idx start from 1
        run_id_str = str(end_point_id)

        # HotFix(Raphael): logging service cross talk
        # Change the handler since each handler need to write to different log files
        try:
            # Remove the existing file handler
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    root_logger.removeHandler(handler)

            # Correct log path: ~/.fedml/fedml-model-server/fedml/logs/fedml-run-$rid-edge-$eid.log
            log_file = os.path.join(ServerConstants.get_log_file_dir(),
                                    f"fedml-run-{run_id_str}-edge-{self.edge_id}.log")

            filehandler = logging.FileHandler(log_file, "a")

            program_prefix = "FedML-Server @device-id-{}".format(self.edge_id)
            formatter = MLOpsFormatter(fmt="[" + program_prefix + "] [%(asctime)s] [%(levelname)s] "
                                                                  "[%(filename)s:%(lineno)d:%(funcName)s] %("
                                                                  "message)s")

            filehandler.setFormatter(formatter)
            root_logger.addHandler(filehandler)
        except Exception as e:
            logging.warning(f"Failed to change the logging handler due to {e}.")

        logging.info("========== callback_deployment_result_message ==========\n")

        # The rolling update and scale out / in operation should not happen at the same time
        assert not ("replica_num_diff" in self.request_json and
                    len(self.request_json["replica_num_diff"]) > 0 and
                    "replica_version_diff" in self.request_json)

        if "replica_version_diff" in self.request_json:
            run_operation = "UPDATE"
        elif "replica_num_diff" in self.request_json and \
                len(self.request_json["replica_num_diff"]) > 0:
            run_operation = "ADD_OR_REMOVE"
        else:
            logging.error(f"Unsupported operation for run id {run_id_str}. and request json "
                          f"{self.request_json}")
            return

        logging.info(f"Endpoint {end_point_id}; Device {device_id}; replica {replica_no}; "
                     f"run_operation {run_operation} model status {model_status}.")

        # OPTIONAL DEBUG PARAMS
        # this_run_controller = self.model_runner_mapping[run_id_str].replica_controller
        # logging.info(f"The current replica controller state is "
        #              f"Total version diff num {this_run_controller.total_replica_version_diff_num}")
        # logging.info(f"self.request_json now {self.request_json}")    # request_json will be deprecated
        # this_run_request_json = self.request_json
        # logging.info(f"self.request_json now {this_run_request_json}")

        # Set redis + sqlite deployment result
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)

        # Deal with different model status
        if model_status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DELETED:
            # remove
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                delete_deployment_result_with_device_id_and_replica_no(
                end_point_id, end_point_name, model_name, device_id, replica_no)
        elif model_status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            # add or update or update-failed-rollback
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_deployment_result(end_point_id, end_point_name,
                                      model_name, model_version,
                                      device_id, payload, replica_no)

            # Note: To display the result in the UI, we need to save successful deployment result to the database
            self.save_deployed_replica_payload(payload_json)
        else:
            if model_status != ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                logging.error(f"Unsupported model status {model_status}.")

            # Avoid endless loop, if the rollback also failed, we should report the failure to the MLOps
            if self.replica_controller.under_rollback or self.is_fresh_endpoint:
                self.send_deployment_status(
                    end_point_id, end_point_name, payload_json["model_name"], "",
                    ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                    message_center=self.message_center)
                return

            # Failure handler, send the rollback message to the worker devices only if it has not been rollback
            if run_operation == "ADD_OR_REMOVE":
                # During Scale out / in,
                # the worker that already been scaled out / in should be sent the rollback message
                rollback_dict = self.replica_controller.rollback_add_or_remove_replica(
                    device_id=device_id, replica_no=replica_no, op_type=run_operation
                )
                self.replica_controller.under_rollback = True

                if rollback_dict is not None and len(rollback_dict) > 0:
                    self.send_deployment_status(
                        end_point_id, end_point_name, payload_json["model_name"], "",
                        ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_ABORTING,
                        message_center=self.message_center)
                    self.send_rollback_add_remove_op(run_id_str, rollback_dict)
                    return
                else:
                    # This is the last worker that failed, so we should continue to "ABORTED" status
                    model_config_parameters = self.request_json["parameters"]
                    inference_port = model_config_parameters.get("server_internal_port",
                                                                 ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
                    inference_port_external = model_config_parameters.get("server_external_port", inference_port)
                    ip = GeneralConstants.get_ip_address(self.request_json)
                    if ip.startswith("http://") or ip.startswith("https://"):
                        model_inference_url = "{}/inference/{}".format(ip, end_point_id)
                    else:
                        model_inference_url = "http://{}:{}/inference/{}".format(ip, inference_port_external,
                                                                                 end_point_id)

                    self.send_deployment_status(
                        end_point_id, end_point_name, payload_json["model_name"], model_inference_url,
                        ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_ABORTED, message_center=self.message_center)

                    # For auto-scaling, should update the state to "DEPLOYED"
                    FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                        update_user_setting_replica_num(end_point_id=end_point_id, state="DEPLOYED")

                    self.replica_controller.under_rollback = False

                    return
            elif run_operation == "UPDATE":
                # Overwrite the json with the rollback version diff
                rollback_version_diff = self.replica_controller.rollback_get_replica_version_diff(
                    device_id_trigger=device_id, replica_no_trigger=replica_no)

                # Change the target version to the start version
                self.replica_controller.rollback_setback_target_replica_version()

                self.request_json["replica_version_diff"] = copy.deepcopy(rollback_version_diff)

                # Send the rollback message to the worker devices
                self.send_rollback_msg(run_id_str)

                # Set the deployment status to ABORTING
                self.send_deployment_status(
                    end_point_id, end_point_name, payload_json["model_name"], "",
                    ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_ABORTING,
                    message_center=self.message_center)

                # TODO(Raphael): Check if resource left not cleaned up
                return

        # Move to the next state (rolling update, finish the deployment, etc.)
        # Notify the replica number controller
        (self.replica_controller.callback_update_curr_replica_num_state(device_id, replica_no, model_status))

        # Notify the replica version controller, which might trigger the next rolling update
        self.send_next_scroll_update_msg(run_id_str, device_id, replica_no)

        # Update the global deployment result mapping
        self.slave_deployment_results_map[str(device_id)] = model_status

        logging.info("callback_deployment_result_message: topic {}, payload {}.".format(topic, payload))

        request_json = self.request_json
        if request_json is None:
            logging.error(f"The endpoint {end_point_id} is no longer running.")
            self.send_deployment_status(
                end_point_id, end_point_name, payload_json["model_name"], "",
                ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                message_center=self.message_center)
            return

        # Wait for all replica-level's result, not device-level
        if (self.replica_controller.is_all_replica_num_reconciled() and
                self.replica_controller.is_all_replica_version_reconciled()):
            """
            When all the devices have finished the add / delete / update operation
            """
            # Generate one unified inference api
            # Note that here we use the gateway port instead of the inference port that is used by the slave device
            model_config_parameters = request_json["parameters"]
            inference_port = model_config_parameters.get("server_internal_port",
                                                         ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
            inference_port_external = model_config_parameters.get("server_external_port", inference_port)
            ip = GeneralConstants.get_ip_address(request_json)

            if ip.startswith("http://") or ip.startswith("https://"):
                model_inference_url = "{}/inference/{}".format(ip, end_point_id)
            else:
                model_inference_url = "http://{}:{}/inference/{}".format(ip, inference_port_external, end_point_id)

            # Send stage: MODEL_DEPLOYMENT_STAGE5 = "StartInferenceIngress"
            self.send_deployment_stages(end_point_id, model_name, model_id,
                                        model_inference_url,
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE5["index"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE5["text"],
                                        "inference url: {}".format(model_inference_url),
                                        message_center=self.message_center)

            # Send the result to MLOps
            if self.deployed_replica_payload is not None:
                payload_json = self.deployed_replica_payload
                model_slave_url = payload_json["model_url"]
                payload_json["model_url"] = model_inference_url
                payload_json["port"] = inference_port_external
                token = FedMLModelCache.get_instance(self.redis_addr, self.redis_port).get_end_point_token(
                    end_point_id, end_point_name, model_name)

                model_metadata = payload_json["model_metadata"]
                model_inputs = model_metadata["inputs"]
                ret_inputs = list()
                if "type" in model_metadata and model_metadata["type"] == "default":
                    payload_json["input_json"] = {"end_point_name": end_point_name,
                                                  "model_name": model_name,
                                                  "token": str(token),
                                                  "inputs": model_inputs,
                                                  "outputs": []}
                    payload_json["output_json"] = model_metadata["outputs"]
                else:
                    raise Exception(f"Unsupported model metadata type {model_metadata['type']}")

                self.send_deployment_results_with_payload(
                    end_point_id, end_point_name, payload_json,
                    self.replica_controller.target_replica_ids)

                payload_json_saved = payload_json
                payload_json_saved["model_slave_url"] = model_slave_url
                FedMLServerDataInterface.get_instance().save_job_result(end_point_id, self.edge_id,
                                                                        json.dumps(payload_json_saved))
            else:
                # Arrive here because only contains remove ops, so we do not need to update the model metadata
                pass

            # For auto-scaling, should update the state to "DEPLOYED"
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                update_user_setting_replica_num(end_point_id=end_point_id, state="DEPLOYED")

            if self.replica_controller.under_rollback:
                # If first time failed (Still might need rollback), then send failed message to the MLOps
                if not (FedMLModelCache.get_instance(self.redis_addr, self.redis_port).
                        get_end_point_activation(end_point_id)):
                    self.send_deployment_status(
                        end_point_id, end_point_name, payload_json["model_name"], "",
                        ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED, message_center=self.message_center)
                else:
                    self.send_deployment_status(
                        end_point_id, end_point_name, payload_json["model_name"], model_inference_url,
                        ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_ABORTED, message_center=self.message_center)

                self.replica_controller.under_rollback = False
            else:
                # Set the end point activation status to True, for scaling out / in and rolling update
                FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                    set_end_point_activation(end_point_id, end_point_name, True)

                self.send_deployment_status(
                    end_point_id, end_point_name, payload_json["model_name"], model_inference_url,
                    ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED, message_center=self.message_center)

            time.sleep(3)
            self.trigger_completed_event()


    def cleanup_runner_process(self, run_id):
        ServerConstants.cleanup_run_process(run_id, not_kill_subprocess=True)

    @staticmethod
    def start_device_inference_gateway(inference_port=ServerConstants.MODEL_INFERENCE_DEFAULT_PORT):
        # start unified inference server
        python_program = get_python_program()
        master_port = os.getenv("FEDML_MASTER_PORT", None)
        if master_port is not None:
            inference_port = int(master_port)
        if not ServerConstants.is_running_on_k8s():
            logging.info(f"start the model inference gateway...")
            inference_gw_cmd = "fedml.computing.scheduler.model_scheduler.device_model_inference:api"
            inference_gateway_pids = RunProcessUtils.get_pid_from_cmd_line(inference_gw_cmd)
            if inference_gateway_pids is None or len(inference_gateway_pids) <= 0:
                cur_dir = os.path.dirname(__file__)
                fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
                inference_gateway_process = ServerConstants.exec_console_with_script(f"{python_program} "
                                                                                     f"-m uvicorn {inference_gw_cmd} "
                                                                                     f"--host 0.0.0.0 "
                                                                                     f"--port {str(inference_port)} "
                                                                                     f"--reload --reload-delay 3 "
                                                                                     f"--reload-dir {fedml_base_dir} "
                                                                                     f"--log-level info",
                                                                                     should_capture_stdout=False,
                                                                                     should_capture_stderr=False)
                return inference_gateway_process
            else:
                return inference_gateway_pids[0]

        return None

    @staticmethod
    def start_device_inference_monitor(
            run_id, end_point_name, model_id, model_name, model_version, check_stopped_event=True,
            redis_addr="localhost", redis_port=6379, redis_password="fedml_default"
    ):
        # start inference monitor server
        # Will report the qps related metrics to the MLOps
        logging.info(f"start the model inference monitor, end point {run_id}, model name {model_name}...")
        run_id_str = str(run_id)
        pip_source_dir = os.path.dirname(__file__)
        monitor_file = os.path.join(pip_source_dir, "device_model_monitor.py")
        python_program = get_python_program()
        running_model_name = ServerConstants.get_running_model_name(end_point_name,
                                                                    model_name, model_version, run_id, model_id)
        monitor_process = ServerConstants.exec_console_with_shell_script_list(
            [python_program, monitor_file, "-v", fedml.get_env_version(), "-ep", run_id_str,
             "-epn", str(end_point_name), "-mi", str(model_id), "-mn", model_name,
             "-mv", model_version, "-iu", "infer_url", "-ra", redis_addr,
             "-rp", str(redis_port), "-rpw", redis_password],
            should_capture_stdout=False, should_capture_stderr=False
        )
        return monitor_process

    @staticmethod
    def stop_device_inference_monitor(run_id, end_point_name, model_id, model_name, model_version):
        # stop inference monitor server
        logging.info(f"stop the model inference monitor, end point {run_id}, model name {model_name}...")
        sys_utils.cleanup_model_monitor_processes(run_id, end_point_name,
                                                  model_id, model_name, model_version)

    @staticmethod
    def recover_inference_and_monitor():
        # noinspection PyBroadException
        try:
            agent_config = dict()
            try:
                agent_config["mqtt_config"], _, _, _ = MLOpsConfigs.fetch_all_configs()
            except Exception as e:
                pass

            history_jobs = FedMLServerDataInterface.get_instance().get_history_jobs()
            for job in history_jobs.job_list:
                if job.running_json is None:
                    continue

                if job.deployment_result == "":
                    continue

                run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
                    model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
                    inference_end_point_id, use_gpu, memory_size, model_version, inference_port = \
                    FedMLDeployMasterJobRunner.parse_model_run_params(json.loads(job.running_json))

                FedMLModelCache.get_instance().set_redis_params()
                is_activated = FedMLModelCache.get_instance().get_end_point_activation(run_id)
                if not is_activated:
                    continue

                FedMLDeployMasterJobRunner.start_device_inference_gateway(inference_port=inference_port)

                FedMLDeployMasterJobRunner.stop_device_inference_monitor(
                    run_id, end_point_name, model_id, model_name, model_version)
                FedMLDeployMasterJobRunner.start_device_inference_monitor(
                    run_id, end_point_name, model_id, model_name, model_version)
        except Exception as e:
            logging.info("recover inference and monitor: {}".format(traceback.format_exc()))

    def send_first_scroll_update_msg(self):
        """
        Replica-level rolling update.
        Delete the record of the replaced device and send the deployment msg to the devices
        """
        if "replica_version_diff" not in self.request_json or self.request_json["replica_version_diff"] is None:
            return []

        first_chunk_dict = self.request_json["replica_version_diff"]

        # Delete the record of the replaced device
        try:
            self.delete_device_replica_info_on_master(
                self.request_json["end_point_id"], self.request_json["end_point_name"],
                self.request_json["model_config"]["model_name"], first_chunk_dict)
        except Exception as e:
            logging.info(f"Exception at send_first_scroll_update_msg {traceback.format_exc()}")

        logging.info(f"Send the first scroll update msg to the device {first_chunk_dict} ")

        # Send the deployment msg to the devices, (we reuse the start_deployment msg)
        for edge_id in first_chunk_dict.keys():
            if edge_id == self.edge_id:
                continue
            # send start deployment request to each device
            self.send_deployment_start_request_to_edge(edge_id, self.request_json)
        return list(first_chunk_dict.keys())

    def send_next_scroll_update_msg(self, run_id_str, device_id, replica_no):
        """
        Send the next scroll update msg to the devices if needed.
        If there is no need for the next scroll update, directly return.
        """
        if replica_no is None:
            return

        replica_controller = self.replica_controller

        if replica_controller.total_replica_version_diff_num == 0:
            return

        if replica_controller.under_rollback:
            replica_controller.intermediate_replica_version[device_id][replica_no] = replica_controller.start_version
            return

        logging.info(f"Curr updating window: {replica_controller.curr_replica_updating_window} "
                     f"Curr version diff num: {replica_controller.total_replica_version_diff_num}")

        replica_controller.callback_update_updating_window(device_id, replica_no)

        # Decide whether to send the next scroll update
        next_chunk_dict = replica_controller.get_next_chunk_devices_replica()

        if next_chunk_dict:
            logging.info(f"The next scroll update for end point {run_id_str} is {next_chunk_dict}")
            # Update curr updating window
            replica_controller.curr_replica_updating_window = copy.deepcopy(next_chunk_dict)

            # Use global deployment result mapping to decide whether to send the next scroll update
            self.request_json["replica_version_diff"] = next_chunk_dict

            # Avoid using the old request_json
            try:
                self.delete_device_replica_info_on_master(
                    self.request_json["end_point_id"],
                    self.request_json["end_point_name"],
                    self.request_json["model_config"]["model_name"],
                    next_chunk_dict)
            except Exception as e:
                logging.info(f"Exception at send_next_scroll_update_msg {traceback.format_exc()}")

            # Send the deployment msg to the devices, (we reuse the start_deployment msg)
            for edge_id in next_chunk_dict.keys():
                if edge_id == self.edge_id:
                    continue
                # send start deployment request to each device
                self.send_deployment_start_request_to_edge(edge_id, self.request_json)
        return

    def send_rollback_msg(self, run_id_str):
        # Avoid using the old request_json
        try:
            self.delete_device_replica_info_on_master(
                self.request_json["end_point_id"],
                self.request_json["end_point_name"],
                self.request_json["model_config"]["model_name"],
                self.request_json["replica_version_diff"])
        except Exception as e:
            logging.info(f"Exception at send_rollback_msg {traceback.format_exc()}")

        # Send the deployment msg to the devices, (we reuse the start_deployment msg)
        for edge_id in self.request_json["replica_version_diff"].keys():
            if edge_id == self.edge_id:
                continue
            # send start deployment request to each device
            self.send_deployment_start_request_to_edge(edge_id, self.request_json)

    def send_rollback_add_remove_op(self, run_id, rollback_replica_dict):
        """
        This method is used when the original add op failed, we need to rollback by delete the existed replicas
        Input example:
        rollback_replica_dict = {'96684': {'curr_num': 2, 'op': 'remove', 'target_num': 1}}
        """
        existed_request_json = self.request_json
        updated_request_json = copy.deepcopy(existed_request_json)

        # Reverse the replica_num_diff
        updated_request_json["replica_num_diff"] = rollback_replica_dict

        self.send_deployment_start_request_to_edges(in_request_json=updated_request_json)

    def delete_device_replica_info_on_master(self, endpoint_id, endpoint_name, model_name, edge_id_replica_no_dict):
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        # Remove the record of the replaced device
        # [Deprecated] deployment status & device info
        # Delete the result in deployment result list in Redis / SQLite
        device_result_list = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_deployment_result_list(endpoint_id, endpoint_name, model_name)

        delete_device_result_list = []
        for device_result in device_result_list:
            device_result_dict = json.loads(device_result)
            if (str(device_result_dict["cache_device_id"]) in edge_id_replica_no_dict.keys() and
                    str(device_result_dict["cache_replica_no"]) in
                    edge_id_replica_no_dict[str(device_result_dict["cache_device_id"])]):
                delete_device_result_list.append(device_result)

        for delete_item in delete_device_result_list:
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port).delete_deployment_result(
                delete_item, endpoint_id, endpoint_name, model_name
            )

        logging.info(f"Deleted the replica record on master: {edge_id_replica_no_dict}")

    def save_deployed_replica_payload(self, payload_json):
        self.deployed_replica_payload = copy.deepcopy(payload_json)

    def get_deployed_replica_payload(self):
        return self.deployed_replica_payload

    def callback_update_curr_replica_num_state(self, changed_device_id, replica_no, op_type):
        if self.replica_controller is not None:
            self.replica_controller.callback_update_curr_replica_num_state(changed_device_id, replica_no, op_type)

    def is_all_replica_num_reconciled(self):
        if self.replica_controller is not None:
            return self.replica_controller.is_all_replica_num_reconciled()

        return False

    def is_all_replica_version_reconciled(self):
        if self.replica_controller is not None:
            return self.replica_controller.is_all_replica_version_reconciled()

        return False

    @staticmethod
    def generate_request_json_with_replica_num_diff(run_id, edge_id, request_json):
        # Replica Controller is per deployment!
        replica_controller = FedMLDeviceReplicaController(edge_id, request_json)
        logging.info(f"Start Diff Replica controller for run {run_id} on edge {edge_id}")

        # Prepare num diff
        run_id_str = str(run_id)
        new_request_with_num_diff = replica_controller.generate_diff_to_request_json()
        request_json = new_request_with_num_diff

        return request_json

    @staticmethod
    def generate_request_json_with_replica_version_diff(run_id, edge_id, request_json):
        # Replica Controller is per deployment!
        replica_controller = FedMLDeviceReplicaController(edge_id, request_json)
        logging.info(f"Start Diff Replica controller for run {run_id} on edge {edge_id}")

        # Prepare version diff
        new_request_with_version_diff = replica_controller.init_first_update_device_replica_mapping()
        request_json = new_request_with_version_diff

        return request_json

    @staticmethod
    def parse_model_run_params(running_json):
        run_id = running_json["end_point_id"]
        end_point_name = running_json["end_point_name"]
        token = running_json["token"]
        user_id = running_json["user_id"]
        user_name = running_json["user_name"]
        device_ids = running_json["device_ids"]
        device_objs = running_json["device_objs"]

        model_config = running_json["model_config"]
        model_name = model_config["model_name"]
        model_id = model_config["model_id"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        inference_engine = model_config.get("inference_engine", 0)
        model_is_from_open = model_config["is_from_open"]
        inference_end_point_id = run_id
        use_gpu = "gpu"  # TODO: Get GPU from device infos
        memory_size = "256m"  # TODO: Get Memory size for each instance
        model_version = model_config["model_version"]
        model_config_parameters = running_json.get("parameters", {})

        inference_port = model_config_parameters.get("server_internal_port",  # Internal port is for the gateway
                                                     ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
        inference_port_external = model_config_parameters.get("server_external_port", inference_port)

        return run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
            model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
            inference_end_point_id, use_gpu, memory_size, model_version, inference_port

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

