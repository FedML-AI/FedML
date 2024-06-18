import fedml

import logging
import os
import shutil
import time
import traceback
import yaml
import datetime
import docker

import requests
import torch
import torch.nn
import tritonclient.http as http_client

import collections.abc

from fedml.computing.scheduler.comm_utils import sys_utils, security_utils
from fedml.computing.scheduler.comm_utils.hardware_utils import HardwareUtil
from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from ..scheduler_core.compute_utils import ComputeUtils
from ..comm_utils.container_utils import ContainerUtils
from .device_http_inference_protocol import FedMLHttpInference

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

no_real_gpu_allocation = None


def request_gpu_ids_on_deployment(edge_id, end_point_id, num_gpus=None, master_device_id=None):
    gpu_ids = None
    client_device_id = os.getenv("FEDML_CURRENT_EDGE_ID")

    if gpu_ids is None:
        cuda_visible_gpu_ids = JobRunnerUtils.get_instance().occupy_gpu_ids(
            end_point_id, num_gpus, client_device_id, inner_id=end_point_id,
            model_master_device_id=master_device_id, model_slave_device_id=edge_id)
        if cuda_visible_gpu_ids is not None:
            gpu_ids = cuda_visible_gpu_ids.split(',')
            gpu_ids = ComputeUtils.map_str_list_to_int_list(gpu_ids)
            logging.info(f"Requested cuda visible gpu ids: {gpu_ids}")

    if gpu_ids is None:
        raise Exception("Failed to request gpu ids!")

    if not torch.cuda.is_available():
        gpu_attach_cmd = ""
    else:
        gpu_id_map = map(lambda x: str(x), gpu_ids)
        gpu_ids_str = ','.join(gpu_id_map)
        gpu_attach_cmd = f"--gpus '\"device={gpu_ids_str}\"'"

    return gpu_ids, gpu_attach_cmd


def start_deployment(end_point_id, end_point_name, model_id, model_version,
                     model_storage_local_path, inference_model_name, inference_engine,
                     infer_host, master_ip, edge_id, master_device_id=None, replica_rank=0,
                     gpu_per_replica=1):
    logging.info("[Worker] Model deployment is starting...")

    # Real gpu per replica (container-level)
    num_gpus = gpu_per_replica
    gpu_ids, gpu_attach_cmd = None, ""

    # Concatenate the model name
    running_model_name = ClientConstants.get_running_model_name(
        end_point_name, inference_model_name, model_version, end_point_id, model_id, edge_id=edge_id)

    # Parse the model config file and get the necessary information for the deployment
    model_config_path = os.path.join(model_storage_local_path, "fedml_model_config.yaml")
    with open(model_config_path, 'r') as file:
        config = yaml.safe_load(file)

        # Resource related
        inference_type = "default"
        use_gpu = config.get('use_gpu', True)
        num_gpus_frm_yml = config.get('num_gpus', None)
        if not use_gpu:
            num_gpus = 0
        else:
            if num_gpus_frm_yml is not None:
                num_gpus = int(num_gpus_frm_yml)
        usr_indicated_wait_time = config.get('deploy_timeout', 900)
        usr_indicated_retry_cnt = max(int(usr_indicated_wait_time) // 10, 1)
        shm_size = config.get('shm_size', None)
        storage_opt = config.get('storage_opt', None)
        tmpfs = config.get('tmpfs', None)
        cpus = config.get('cpus', None)
        if cpus is not None:
            cpus = int(cpus)
        memory = config.get('memory', None)

        inference_image_name = config.get('inference_image_name',
                                          ClientConstants.INFERENCE_SERVER_CUSTOME_IMAGE)
        image_pull_policy = config.get('image_pull_policy', SchedulerConstants.IMAGE_PULL_POLICY_IF_NOT_PRESENT)

        # Source code dir, bootstrap dir, data cache dir
        src_code_dir = os.path.join(model_storage_local_path, config.get('source_code_dir', ""))

        # Get the bootstrap and job commands inside the yaml file
        bootstrap_cmds_str_frm_yaml = config.get('bootstrap', "")
        job_cmds_str_frm_yaml = config.get('job', "")

        if bootstrap_cmds_str_frm_yaml != "" or job_cmds_str_frm_yaml != "":
            auto_gen_bootstrap_file_name = "fedml-deploy-bootstrap-entry-auto-gen.sh"
            src_bootstrap_file_path = os.path.join(model_storage_local_path, auto_gen_bootstrap_file_name)
            with open(src_bootstrap_file_path, 'w') as f:
                f.write("cd /home/fedml/models_serving/\n")
                f.write(bootstrap_cmds_str_frm_yaml)
                f.write("\n")
                f.write("cd /home/fedml/models_serving/\n")
                f.write(job_cmds_str_frm_yaml)
        else:
            src_bootstrap_file_path = ""

        data_cache_dir_input = config.get('data_cache_dir', "")
        request_input_example = config.get('request_input_example', None)
        extra_envs = config.get('environment_variables', None)

        # Serving dir inside docker
        dst_model_serving_dir = "/home/fedml/models_serving"
        relative_entry = config.get('entry_point')
        if src_bootstrap_file_path != "":
            dst_bootstrap_dir = os.path.join(dst_model_serving_dir, auto_gen_bootstrap_file_name)
        else:
            dst_bootstrap_dir = ""

        # If using customized image, then bootstrap + job will be the entry point
        enable_custom_image = config.get("enable_custom_image", False)
        # inference_type = "custom"
        customized_image_entry_cmd = \
            "/bin/bash /home/fedml/models_serving/fedml-deploy-bootstrap-entry-auto-gen.sh"

        docker_registry_user_name = config.get("docker_registry_user_name", "")
        docker_registry_user_password = config.get("docker_registry_user_password", "")
        docker_registry = config.get("docker_registry", "")

        port_inside_container = int(config.get("port", 2345))

    # Request the GPU ids for the deployment
    if num_gpus > 0:
        gpu_ids, gpu_attach_cmd = request_gpu_ids_on_deployment(
            edge_id, end_point_id, num_gpus=num_gpus, master_device_id=master_device_id)

        # set replica and their gpu ids
        FedMLModelCache.get_instance().set_redis_params()
        FedMLModelCache.get_instance().set_replica_gpu_ids(
            end_point_id, end_point_name, inference_model_name, edge_id, replica_rank+1, gpu_ids)
    logging.info("GPU ids allocated: {}".format(gpu_ids))

    # Create the model serving dir if not exists
    model_serving_dir = ClientConstants.get_model_serving_dir()
    if not os.path.exists(model_serving_dir):
        os.makedirs(model_serving_dir, exist_ok=True)

    if inference_engine != ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEFAULT:
        raise Exception(f"inference engine {inference_engine} is not supported")

    # Get the master device id
    logging.info(f"master ip: {master_ip}, worker ip: {infer_host}")
    if infer_host == master_ip:
        logging.info("infer_host is the same as master ip, will use 127.0.0.1 to avoid firewall issue")
        infer_host = "127.0.0.1"

    try:
        client = docker.from_env()
        if enable_custom_image and docker_registry_user_name != "" and docker_registry_user_password != "" \
                and docker_registry != "":
            client.login(username=docker_registry_user_name, password=docker_registry_user_password,
                         registry=docker_registry)
    except Exception:
        logging.error("Failed to connect to the docker daemon, please ensure that you have "
                      "installed Docker Desktop or Docker Engine, and the docker is running")
        return "", "", None, None, None

    container_prefix = ("{}".format(ClientConstants.FEDML_DEFAULT_SERVER_CONTAINER_NAME_PREFIX) + "__" +
                        security_utils.get_content_hash(running_model_name))

    default_server_container_name = container_prefix + "__" + str(replica_rank)

    try:
        exist_container_obj = client.containers.get(default_server_container_name)
    except docker.errors.NotFound:
        exist_container_obj = None
    except docker.errors.APIError:
        raise Exception("Failed to get the container object")

    # Pull the inference image
    logging.info(f"Start pulling the inference image {inference_image_name}... with policy {image_pull_policy}")
    ContainerUtils.get_instance().pull_image_with_policy(image_pull_policy, inference_image_name)

    volumes = []
    binds = {}
    environment = {}

    # data_cache_dir mounting
    if isinstance(data_cache_dir_input, str):
        # In this case, we mount to the same folder, if it has ~, we replace it with /home/fedml
        src_data_cache_dir, dst_data_cache_dir = "", ""
        if data_cache_dir_input != "":
            if data_cache_dir_input[0] == "~":
                src_data_cache_dir = os.path.expanduser(data_cache_dir_input)
                dst_data_cache_dir = data_cache_dir_input.replace("~", "/home/fedml")
            else:
                # check if the data_cache_dir is a relative path
                if data_cache_dir_input[0] != "/":
                    raise "data_cache_dir_input has to be an absolute path or start with ~"
                else:
                    src_data_cache_dir = data_cache_dir_input
                    dst_data_cache_dir = data_cache_dir_input
            logging.info(f"src_data_cache_dir: {src_data_cache_dir}, dst_data_cache_dir: {dst_data_cache_dir}")

            if type(src_data_cache_dir) == str and src_data_cache_dir != "":
                logging.info("Start copying the data cache to the container...")
                if os.path.exists(src_data_cache_dir):
                    volumes.append(src_data_cache_dir)
                    binds[src_data_cache_dir] = {
                        "bind": dst_data_cache_dir,
                        "mode": "rw"
                    }
                    environment["DATA_CACHE_FOLDER"] = dst_data_cache_dir
    elif isinstance(data_cache_dir_input, dict):
        for k, v in data_cache_dir_input.items():
            if os.path.exists(k):
                volumes.append(v)
                binds[k] = {
                    "bind": v,
                    "mode": "rw"
                }
            else:
                logging.warning(f"{k} does not exist, skip mounting it to the container")
        logging.info(f"Data cache mount: {volumes}, {binds}")
    else:
        logging.warning("data_cache_dir_input is not a string or a dictionary, skip mounting it to the container")

    # Default mounting
    if not enable_custom_image or (enable_custom_image and relative_entry != ""):
        logging.info("Start copying the source code to the container...")
        volumes.append(src_code_dir)
        binds[src_code_dir] = {
            "bind": dst_model_serving_dir,
            "mode": "rw"
        }
        environment["MAIN_ENTRY"] = relative_entry

    host_config_dict = {
        "binds": binds,
        "port_bindings": {
            port_inside_container: None
        },
        "shm_size": shm_size,
        "storage_opt": storage_opt,
        "tmpfs": tmpfs,
        "cpu_count": cpus,
        "mem_limit": memory
    }

    # Allocate the GPU
    # TODO: Make sure no competition for each replica in a single deployment
    if exist_container_obj is not None:
        client.api.remove_container(exist_container_obj.id, v=True, force=True)
    device_mapping = {}
    if no_real_gpu_allocation is not None:
        use_gpu = not no_real_gpu_allocation
    if use_gpu:
        logging.info("Number of GPUs: {}".format(num_gpus))
        device_mapping = HardwareUtil.get_docker_gpu_device_mapping(gpu_ids, num_gpus)
    logging.info(f"device_mapping: {device_mapping}")

    if device_mapping:
        host_config_dict.update(device_mapping)

    # Environment variables
    if not enable_custom_image:
        # For some image, the default user is root. Unified to fedml.
        environment["HOME"] = "/home/fedml"
    environment["BOOTSTRAP_DIR"] = dst_bootstrap_dir
    environment["FEDML_CURRENT_RUN_ID"] = end_point_id
    environment["FEDML_CURRENT_EDGE_ID"] = edge_id
    environment["FEDML_REPLICA_RANK"] = replica_rank
    environment["FEDML_CURRENT_VERSION"] = fedml.get_env_version()
    environment["FEDML_ENV_VERSION"] = fedml.get_env_version()
    environment["FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_HOST"] = fedml.get_local_on_premise_platform_host()
    environment["FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_PORT"] = fedml.get_local_on_premise_platform_port()

    if extra_envs is not None:
        for key in extra_envs:
            environment[key] = extra_envs[key]

    # Create the container
    try:
        host_config = client.api.create_host_config(**host_config_dict)
        new_container = client.api.create_container(
            image=inference_image_name,
            name=default_server_container_name,
            volumes=volumes,
            ports=[port_inside_container],  # port open inside the container
            environment=environment,
            host_config=host_config,
            detach=True,
            command=customized_image_entry_cmd if enable_custom_image else None,
            entrypoint=customized_image_entry_cmd if enable_custom_image else None
        )
        client.api.start(container=new_container.get("Id"))
    except Exception as e:
        logging.error(f"Failed to create the container with exception {e}, traceback : {traceback.format_exc()}")
        return "", "", None, None, None

    # Get the port allocation
    cnt = 0
    while True:
        cnt += 1
        try:
            # Find the random port
            port_info = client.api.port(new_container.get("Id"), port_inside_container)
            inference_http_port = port_info[0]["HostPort"]
            logging.info("host port allocated: {}".format(inference_http_port))
            break
        except:
            if cnt >= 5:
                raise Exception("Failed to get the port allocation")
            time.sleep(3)

    # Logging the info from the container when starting
    log_deployment_output(end_point_id, model_id, default_server_container_name,
                          ClientConstants.CMD_TYPE_RUN_DEFAULT_SERVER,
                          inference_model_name, inference_engine, inference_http_port, inference_type,
                          retry_interval=10, deploy_attempt_threshold=usr_indicated_retry_cnt,
                          request_input_example=request_input_example, infer_host=infer_host,
                          enable_custom_image=enable_custom_image)

    # Return the running model name and the inference output url
    inference_output_url, running_model_version, ret_model_metadata, ret_model_config = \
        check_container_readiness(inference_http_port=inference_http_port, infer_host=infer_host,
                                  request_input_example=request_input_example)

    if inference_output_url == "":
        return running_model_name, "", None, None, None

    # Successfully get the result from the container
    model_metadata = ret_model_metadata
    logging.info(f"[Worker][Replica{replica_rank}] Model deployment is successful with inference_output_url: "
                 f"{inference_output_url}, model_metadata: {model_metadata}, model_config: {ret_model_config}")

    return running_model_name, inference_output_url, model_version, model_metadata, ret_model_config


def build_inference_req(end_point_name, model_name, token, in_model_metadata):
    model_inputs = in_model_metadata["inputs"]
    ret_inputs = list()

    for input_item in model_inputs:
        ret_item = input_item
        shape = ret_item["shape"]
        data_type = ret_item["datatype"]
        if ClientConstants.MODEL_DATA_TYPE_MAPPING[data_type] == ClientConstants.MODEL_DATA_TYPE_INT:
            for i in range(len(shape)):
                if shape[i] == -1:  # if input shape is dynamic, we set a default value 1
                    shape[i] = 1
            ret_item["data"] = torch.randint(0, 1, shape).tolist()
        else:
            for i in range(len(shape)):
                if shape[i] == -1:  # if input shape is dynamic, we set a default value 1
                    shape[i] = 1
            ret_item["data"] = torch.zeros(shape).tolist()
        ret_inputs.append(ret_item)

    input_json = {"end_point_name": end_point_name,
                  "model_name": model_name,
                  "token": str(token),
                  "inputs": ret_inputs,
                  "outputs": in_model_metadata["outputs"]}
    output_json = in_model_metadata["outputs"]

    return input_json, output_json


def should_exit_logs(end_point_id, model_id, cmd_type, model_name, inference_engine, inference_port,
                     inference_type="default", request_input_example=None, infer_host="127.0.0.1",
                     enable_custom_image=False):
    if cmd_type == ClientConstants.CMD_TYPE_RUN_DEFAULT_SERVER:
        # TODO: Exited Quickly if the container is Exited or Removed
        # If the container has exited, return True, means we should exit the logs
        try:
            inference_output_url, model_version, model_metadata, model_config = \
                check_container_readiness(inference_http_port=inference_port, infer_host=infer_host,
                                          request_input_example=request_input_example)
            if inference_output_url != "":
                logging.info("Log test for deploying model successfully, inference url: {}, "
                             "model metadata: {}, model config: {}".
                             format(inference_output_url, model_metadata, model_config))
                return True
        except Exception as e:
            pass

        return False
    else:
        logging.error("Unknown cmd type: {}".format(cmd_type))
        return False


def log_deployment_output(end_point_id, model_id, cmd_container_name, cmd_type,
                          inference_model_name, inference_engine,
                          inference_http_port, inference_type="default",
                          retry_interval=10, deploy_attempt_threshold=10,
                          request_input_example=None, infer_host="127.0.0.1",
                          enable_custom_image=False):
    deploy_attempt = 0
    last_log_time = datetime.datetime.now()

    while True:
        if not ClientConstants.is_running_on_k8s():
            logging.info(f"Attempt: {deploy_attempt} / {deploy_attempt_threshold} ...")

            try:
                client = docker.from_env()
            except Exception:
                logging.error("Failed to connect to the docker daemon, please ensure that you have "
                              "installed Docker Desktop or Docker Engine, and the docker is running")
                break

            try:
                container_obj = client.containers.get(cmd_container_name)
            except docker.errors.NotFound:
                logging.error("Container {} not found".format(cmd_container_name))
                break
            except docker.errors.APIError:
                logging.error("The API cannot be accessed")
                break

            if container_obj is not None:
                out_logs, err_logs = None, None
                try:
                    out_logs = container_obj.logs(stdout=True, stderr=False, stream=False, follow=False,
                                                  since=last_log_time)
                    err_logs = container_obj.logs(stdout=False, stderr=True, stream=False, follow=False,
                                                  since=last_log_time)
                except Exception as e:
                    logging.error(f"Failed to get the logs from the container with exception {e}")
                    pass

                last_log_time = datetime.datetime.now()

                if err_logs is not None:
                    err_logs = sys_utils.decode_our_err_result(err_logs)
                    if len(err_logs) > 0:
                        logging.error(f"{format(err_logs)}")

                if out_logs is not None:
                    out_logs = sys_utils.decode_our_err_result(out_logs)
                    if len(out_logs) > 0:
                        logging.info(f"{format(out_logs)}")

                if container_obj.status == "exited":
                    logging.info("Container {} has exited, automatically remove it".format(cmd_container_name))

                    # Save the failed log into ~/.fedml/fedml-model-client/fedml/logs/failed_logs/
                    # $run_id/$container_name.log
                    try:
                        parent_dir = os.path.join(ClientConstants.get_deploy_failed_log_dir())
                        os.makedirs(parent_dir, exist_ok=True)
                        error_logs_dir = os.path.join(ClientConstants.get_deploy_failed_log_dir(), str(end_point_id))
                        os.makedirs(error_logs_dir, exist_ok=True)
                        error_log_file = os.path.join(error_logs_dir, f"{cmd_container_name}.log")
                        with open(error_log_file, "w") as f:
                            f.write(f"Container {cmd_container_name} has exited\n")
                            f.write(f"Error logs: {err_logs}\n")
                            f.write(f"Output logs: {out_logs}\n")
                    except Exception as e:
                        logging.error(f"Failed to save the error logs with exception {e}")

                    client.api.remove_container(container_obj.id, v=True, force=True)
                    break

        # should_exit_logs will ping the inference container
        # return True if ready
        if should_exit_logs(end_point_id, model_id, cmd_type, inference_model_name, inference_engine,
                            inference_http_port, inference_type, request_input_example,
                            infer_host, enable_custom_image=enable_custom_image):
            break

        # Not yet ready, retry
        deploy_attempt += 1
        if deploy_attempt >= deploy_attempt_threshold:
            logging.error(f"Model {inference_model_name} deploy reached max attempt {deploy_attempt_threshold}, "
                          f"exiting the deployment...")

            try:
                client = docker.from_env()
                container_obj = client.containers.get(cmd_container_name)
                if container_obj is not None:
                    client.api.remove_container(container_obj.id, v=True, force=True)
            except Exception as e:
                logging.error(f"Failed to remove the container with exception {e}")
            break

        logging.info(f"Model {inference_model_name} not yet ready, retry in {retry_interval} seconds...")
        time.sleep(retry_interval)


def is_client_inference_container_ready(infer_url_host, inference_http_port, readiness_check_type="default",
                                        readiness_check_cmd=None, request_input_example=None):

    if readiness_check_type == "default":
        default_client_container_ready_url = "http://{}:{}/ready".format("0.0.0.0", inference_http_port)
        response = None
        try:
            response = requests.get(default_client_container_ready_url)
        except:
            pass
        if not response or response.status_code != 200:
            return "", "", {}, {}

        # Construct the model metadata (input and output)
        model_metadata = {}
        if request_input_example is not None and len(request_input_example) > 0:
            model_metadata["inputs"] = request_input_example
        else:
            model_metadata["inputs"] = {"text": "What is a good cure for hiccups?"}
        model_metadata["outputs"] = []
        model_metadata["type"] = "default"

        return "http://{}:{}/predict".format(infer_url_host, inference_http_port), None, model_metadata, None
    else:
        # TODO(Raphael): Support arbitrary readiness check command
        logging.error(f"Unknown readiness check type: {readiness_check_type}")
        return "", "", {}, {}


def check_container_readiness(inference_http_port, infer_host="127.0.0.1", request_input_example=None,
                              readiness_check_type="default", readiness_check_cmd=None):
    response_from_client_container = is_client_inference_container_ready(
        infer_host, inference_http_port, readiness_check_type, readiness_check_cmd,
        request_input_example=request_input_example)

    return response_from_client_container


def run_http_inference_with_curl_request(inference_url, inference_input_list, inference_output_list,
                                         inference_type="default", engine_type="default", timeout=None):
    return FedMLHttpInference.run_http_inference_with_curl_request(
        inference_url, inference_input_list, inference_output_list,
        inference_type=inference_type, engine_type=engine_type, timeout=timeout)


if __name__ == "__main__":
    pass
