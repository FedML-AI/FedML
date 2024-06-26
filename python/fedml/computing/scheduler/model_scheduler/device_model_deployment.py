import fedml

import logging
import os
import time
import traceback
import yaml
import datetime
import docker

import requests
import torch
import torch.nn

import collections.abc

from fedml.computing.scheduler.comm_utils import sys_utils, security_utils
from fedml.computing.scheduler.comm_utils.hardware_utils import HardwareUtil
from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
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
                     gpu_per_replica=1, request_json=None):
    if request_json is None:
        request_json = dict()
    logging.info("[Worker] Model deployment is starting...")

    # Real gpu per replica (container-level)
    num_gpus = gpu_per_replica
    gpu_ids, gpu_attach_cmd = None, ""

    # Concatenate the full model name
    running_model_name = ClientConstants.get_running_model_name(
        end_point_name, inference_model_name, model_version, end_point_id, model_id, edge_id=edge_id)

    # Parse the model config file
    model_config_path = os.path.join(model_storage_local_path, "fedml_model_config.yaml")
    with open(model_config_path, 'r') as file:
        config = yaml.safe_load(file)
        inference_type = "default"

        # Resource related
        use_gpu, num_gpus, shm_size, storage_opt, tmpfs, cpus, memory, port_inside_container = \
            parse_resource_related_config(config, gpu_per_replica)

        # Image related
        inference_image_name, image_pull_policy, registry_name, registry_provider, \
            registry_user_name, registry_user_password = parse_image_registry_related_config(config)

        # Service app related
        dst_bootstrap_dir, dst_model_serving_dir, relative_entry_fedml_format, enable_serverless_container, \
            customized_image_entry_cmd, customized_readiness_check, customized_liveliness_check, customized_uri = \
            handle_container_service_app(config, model_storage_local_path)

        # Storage related
        src_code_dir = os.path.join(model_storage_local_path, config.get('source_code_dir', ""))
        data_cache_dir_input = config.get('data_cache_dir', "")
        usr_customized_mount_rule = config.get(ClientConstants.CUSTOMIZED_VOLUMES_MOUNT_KEY, None)

        # Others
        extra_envs = config.get('environment_variables', None)
        usr_indicated_wait_time = config.get(ClientConstants.DEPLOY_TIMEOUT_SEC_KEY,
                                             config.get("deploy_timeout", ClientConstants.DEPLOY_TIMEOUT_SEC_DEFAULT))
        usr_indicated_retry_cnt = max(int(usr_indicated_wait_time) // 10, 1)
        request_input_example = config.get('request_input_example', None)

    # Parameter's check
    if inference_engine != ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEFAULT:
        raise Exception(f"inference engine {inference_engine} is not supported")

    # Request the GPU
    if num_gpus > 0:
        gpu_ids, gpu_attach_cmd = request_gpu_ids_on_deployment(
            edge_id, end_point_id, num_gpus=num_gpus, master_device_id=master_device_id)
        FedMLModelCache.get_instance().set_redis_params()
        FedMLModelCache.get_instance().set_replica_gpu_ids(
            end_point_id, end_point_name, inference_model_name, edge_id, replica_rank+1, gpu_ids)
    logging.info("GPU ids allocated: {}".format(gpu_ids))

    # Create the model serving dir if not exists
    model_serving_dir = ClientConstants.get_model_serving_dir()
    if not os.path.exists(model_serving_dir):
        os.makedirs(model_serving_dir, exist_ok=True)

    # Determine whether to report public ip or localhost
    if infer_host == master_ip:
        logging.info("infer_host is the same as master ip, will use 127.0.0.1 to avoid firewall issue")
        infer_host = "127.0.0.1"
    else:
        logging.info("Master and worker are located in different machines, will use the public ip for inference")

    # Init container interface client
    try:
        client = docker.from_env()
        if registry_provider == "Docker" and registry_user_name != "" and registry_user_password != "" \
                and registry_name != "":
            client.login(username=registry_user_name, password=registry_user_password,
                         registry=registry_name)
    except Exception:
        logging.error("Failed to connect to the docker daemon, please ensure that you have "
                      "installed Docker Desktop or Docker Engine, and the docker is running")
        return "", "", None, None, None

    # Pull the inference image
    logging.info(f"Start pulling the inference image {inference_image_name}... with policy {image_pull_policy}")
    ContainerUtils.get_instance().pull_image_with_policy(image_pull_policy, inference_image_name)

    # Remove if the container exists
    container_prefix = ("{}".format(ClientConstants.FEDML_DEFAULT_SERVER_CONTAINER_NAME_PREFIX) + "__" +
                        security_utils.get_content_hash(running_model_name))
    default_server_container_name = container_prefix + "__" + str(replica_rank)
    try:
        exist_container_obj = client.containers.get(default_server_container_name)
    except docker.errors.NotFound:
        exist_container_obj = None
    except docker.errors.APIError:
        raise Exception("Failed to get the container object")
    # Allocate the GPU
    # TODO: Make sure no competition for each replica in a single deployment
    if exist_container_obj is not None:
        client.api.remove_container(exist_container_obj.id, v=True, force=True)

    # Build host config
    volumes = []
    binds = {}
    environment = {}

    # Handle the union volume mount
    _handle_union_volume_mount(binds, volumes, environment, data_cache_dir_input)

    # Handle the default volume mount
    handle_volume_mount(volumes, binds, environment, relative_entry_fedml_format, src_code_dir,
                        dst_model_serving_dir, usr_customized_mount_rule, host_workspace_root=model_storage_local_path)

    # Host config
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

    device_mapping = {}
    if no_real_gpu_allocation is not None:
        use_gpu = not no_real_gpu_allocation
    if use_gpu:
        logging.info("Number of GPUs: {}".format(num_gpus))
        device_mapping = HardwareUtil.get_docker_gpu_device_mapping(gpu_ids, num_gpus)
    logging.info(f"device_mapping: {device_mapping}")

    if device_mapping:
        host_config_dict.update(device_mapping)

    # Handle the environment variables
    handle_env_vars(environment, relative_entry_fedml_format, extra_envs, dst_bootstrap_dir,
                    end_point_id, edge_id, replica_rank, request_json)

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
            command=customized_image_entry_cmd,
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

    # Logging the info from the container when initializing
    log_deployment_output(end_point_id, model_id, default_server_container_name,
                          ClientConstants.CMD_TYPE_RUN_DEFAULT_SERVER,
                          inference_model_name, inference_engine, inference_http_port, inference_type,
                          retry_interval=10, deploy_attempt_threshold=usr_indicated_retry_cnt,
                          request_input_example=request_input_example, infer_host=infer_host,
                          readiness_check=customized_readiness_check)

    # Return the running model name and the inference output url
    inference_output_url, running_model_version, ret_model_metadata, ret_model_config = \
        check_container_readiness(inference_http_port=inference_http_port, infer_host=infer_host,
                                  readiness_check=customized_readiness_check,
                                  request_input_example=request_input_example,
                                  customized_uri=customized_uri)

    if inference_output_url == "":
        return running_model_name, "", None, None, None

    # Successfully get the result from the container
    model_metadata = ret_model_metadata
    model_metadata["liveliness_check"] = customized_liveliness_check
    model_metadata["readiness_check"] = customized_readiness_check
    model_metadata[ClientConstants.ENABLE_SERVERLESS_CONTAINER_KEY] = enable_serverless_container
    logging.info(f"[Worker][Replica{replica_rank}] Model deployment is successful with inference_output_url: "
                 f"{inference_output_url}, model_metadata: {model_metadata}, model_config: {ret_model_config}")

    return running_model_name, inference_output_url, model_version, model_metadata, ret_model_config


def should_exit_logs(end_point_id, model_id, cmd_type, model_name, inference_engine, inference_port,
                     inference_type="default", request_input_example=None, infer_host="127.0.0.1",
                     readiness_check=ClientConstants.READINESS_PROBE_DEFAULT):
    if cmd_type == ClientConstants.CMD_TYPE_RUN_DEFAULT_SERVER:
        # TODO: Exited Quickly if the container is Exited or Removed
        # If the container has exited, return True, means we should exit the logs
        try:
            inference_output_url, model_version, model_metadata, model_config = \
                check_container_readiness(inference_http_port=inference_port, infer_host=infer_host,
                                          readiness_check=readiness_check,
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
                          readiness_check=ClientConstants.READINESS_PROBE_DEFAULT):
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

        # should_exit_logs will ping the inference container, return True if ready
        if should_exit_logs(end_point_id, model_id, cmd_type, inference_model_name, inference_engine,
                            inference_http_port, inference_type, request_input_example,
                            infer_host, readiness_check=readiness_check):
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


def parse_resource_related_config(config, gpu_num_frm_platform=0):
    use_gpu = config.get('use_gpu', True)
    num_gpus_frm_yml = config.get('num_gpus', None)

    num_gpus = gpu_num_frm_platform
    # Priority: num_gpus from yaml > num_gpus from platform
    if use_gpu:
        if num_gpus_frm_yml is not None:
            num_gpus = int(num_gpus_frm_yml)
    else:
        num_gpus = 0

    shm_size = config.get('shm_size', None)
    storage_opt = config.get('storage_opt', None)
    tmpfs = config.get('tmpfs', None)
    cpus = config.get('cpus', None)
    if cpus is not None:
        cpus = int(cpus)
    memory = config.get('memory', None)
    port_inside_container = int(config.get("port", 2345))

    return use_gpu, num_gpus, shm_size, storage_opt, tmpfs, cpus, memory, port_inside_container


def parse_image_registry_related_config(config):
    inference_image_name = config.get('inference_image_name', ClientConstants.INFERENCE_SERVER_CUSTOME_IMAGE)
    image_pull_policy = config.get('image_pull_policy', SchedulerConstants.IMAGE_PULL_POLICY_IF_NOT_PRESENT)

    # Optional
    registry_specs = config.get('registry_specs', {})
    registry_name = registry_specs.get("docker_registry_user_name", "")
    registry_provider = registry_specs.get("registry_provider", "")
    registry_user_name = config.get("registry_user_name", "")
    registry_user_password = config.get("registry_user_password", "")

    return (inference_image_name, image_pull_policy, registry_name, registry_provider,
            registry_user_name, registry_user_password)


def is_client_inference_container_ready(infer_url_host, inference_http_port,
                                        readiness_check=ClientConstants.READINESS_PROBE_DEFAULT,
                                        request_input_example=None, container_id=None, customized_uri=None):
    # Construct the model metadata (input and output)
    model_metadata = {}
    if request_input_example is not None and len(request_input_example) > 0:
        model_metadata["inputs"] = request_input_example
    else:
        model_metadata["inputs"] = {"text": "What is a good cure for hiccups?"}
    model_metadata["outputs"] = []
    model_metadata["type"] = "default"

    # Check the readiness of the container
    if readiness_check == ClientConstants.READINESS_PROBE_DEFAULT:
        default_client_container_ready_url = "http://{}:{}/ready".format("0.0.0.0", inference_http_port)
        response = None
        try:
            response = requests.get(default_client_container_ready_url)
        except:
            pass
        if not response or response.status_code != 200:
            return "", "", {}, {}

        return "http://{}:{}/predict".format(infer_url_host, inference_http_port), None, model_metadata, None
    else:
        if not isinstance(readiness_check, dict):
            logging.error(f"Unknown readiness check type: {readiness_check}")
            return "", "", {}, {}

        if "httpGet" in readiness_check:
            if "path" in readiness_check["httpGet"]:
                check_path = readiness_check["httpGet"]["path"]
                if not isinstance(check_path, str):
                    logging.error(f"Invalid path type: {check_path}, expected str")
                    return "", "", {}, {}
                else:
                    if not check_path.startswith("/"):
                        check_path = "/" + check_path
                response = None
                try:
                    response = requests.get(f"http://{infer_url_host}:{inference_http_port}{check_path}")
                except:
                    pass
                if not response or response.status_code != 200:
                    return "", "", {}, {}
            else:
                logging.error("'path' is not specified in httpGet readiness check")
                return "", "", {}, {}
        elif "exec" in readiness_check:
            # TODO(raphael): Support arbitrary readiness check command by using container id and docker exec
            pass
        else:
            # Ref K8S, if no readiness check, we assume the container is ready immediately
            pass

        # Construct the customized URI
        path = ""
        if customized_uri is not None:
            if "httpPost" in customized_uri and "path" in customized_uri["httpPost"]:
                path = customized_uri["httpPost"]["path"]
                if not isinstance(path, str):
                    logging.error(f"Invalid path type: {path}, expected str")
                    return "", "", {}, {}
                else:
                    if not path.startswith("/"):
                        path = "/" + path
            # TODO(raphael): Finalized more customized URI types
        readiness_check_url = f"http://{infer_url_host}:{inference_http_port}{path}"

        return readiness_check_url, None, model_metadata, None


def _handle_union_volume_mount(binds, volumes, environment, data_cache_dir_input=None):
    """
    Private: data_cache_dir is the union folder on host machine, which will be shard across different containers,
    the control of this folder should be handled by the platform.
    """
    if isinstance(data_cache_dir_input, str):
        # In this case, we mount to the same folder, if it has ~, we replace it with /home/fedml
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

            if isinstance(src_data_cache_dir, str) and src_data_cache_dir != "":
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
        logging.info("data_cache_dir_input is not a string or a dictionary, skip mounting it to the container")


def handle_volume_mount(volumes, binds, environment, relative_entry_fedml_format="", src_code_dir="",
                        dst_model_serving_dir="", customized_volumes_mount_rule=None, host_workspace_root=""):
    # If fedml format entry point is specified, inject the source code, e.g., main.py (FedMLPredictor inside)
    if relative_entry_fedml_format != "":
        logging.info("Using FedML format entry point, mounting the source code...")
        volumes.append(src_code_dir)
        binds[src_code_dir] = {
            "bind": dst_model_serving_dir,
            "mode": "rw"
        }
        environment["MAIN_ENTRY"] = relative_entry_fedml_format
        return  # The reason we return here is that we don't need to mount the source code again

    # If customized volume mount rule is specified, just follow the mount rule
    """
    e.g.,
    volumes:
      - workspace_path: "./model_repository"
        mount_path: "/repo_inside_container"
    """
    mount_list = []
    if not isinstance(customized_volumes_mount_rule, list):
        if not isinstance(customized_volumes_mount_rule, dict):
            logging.warning("customized_volumes_mount_rule is not a list or a dictionary, "
                            "skip mounting it to the container")
            return

        # transform the dict to list
        for k, v in customized_volumes_mount_rule.items():
            mount_list.append({ClientConstants.CUSTOMIZED_VOLUMES_PATH_FROM_WORKSPACE_KEY: k,
                               ClientConstants.CUSTOMIZED_VOLUMES_PATH_FROM_CONTAINER_KEY: v})
    else:
        mount_list = customized_volumes_mount_rule if customized_volumes_mount_rule is not None else []

    for mount in mount_list:
        workspace_relative_path = mount[ClientConstants.CUSTOMIZED_VOLUMES_PATH_FROM_WORKSPACE_KEY]
        mount_path = mount[ClientConstants.CUSTOMIZED_VOLUMES_PATH_FROM_CONTAINER_KEY]

        workspace_path = os.path.join(host_workspace_root, workspace_relative_path)
        if os.path.exists(workspace_path):
            volumes.append(workspace_path)
            binds[workspace_path] = {
                "bind": mount_path,
                "mode": "rw"
            }
        else:
            logging.warning(f"{workspace_path} does not exist, skip mounting it to the container")


def handle_container_service_app(config, model_storage_local_path):
    # Bootstrap, job and entrypoint related
    dst_model_serving_dir = "/home/fedml/models_serving"
    bootstrap_cmds_str_frm_yaml = config.get('bootstrap', "")
    job_cmds_str_frm_yaml = config.get('job', "")

    auto_gen_bootstrap_file_name = "fedml-deploy-bootstrap-entry-auto-gen.sh"
    if bootstrap_cmds_str_frm_yaml != "" or job_cmds_str_frm_yaml != "":
        src_bootstrap_file_path = os.path.join(model_storage_local_path, auto_gen_bootstrap_file_name)
        with open(src_bootstrap_file_path, 'w') as f:
            f.write("cd /home/fedml/models_serving/\n")
            f.write(bootstrap_cmds_str_frm_yaml)
            f.write("\n")
            f.write("cd /home/fedml/models_serving/\n")
            f.write(job_cmds_str_frm_yaml)
    else:
        src_bootstrap_file_path = ""

    if src_bootstrap_file_path != "":
        dst_bootstrap_dir = os.path.join(dst_model_serving_dir, auto_gen_bootstrap_file_name)
    else:
        dst_bootstrap_dir = ""

    # If the entry point is in fedml format (e.g., "main.py")
    relative_entry_fedml_format = config.get('entry_point', "")

    # User indicate either fedml format python main entry filename or entry command
    enable_serverless_container = config.get(ClientConstants.ENABLE_SERVERLESS_CONTAINER_KEY, False)
    customized_image_entry_cmd = config.get('container_run_command', None)  # Could be str or list
    customized_readiness_check = config.get('readiness_probe', ClientConstants.READINESS_PROBE_DEFAULT)
    customized_liveliness_check = config.get('liveness_probe', ClientConstants.LIVENESS_PROBE_DEFAULT)
    customized_uri = config.get(ClientConstants.CUSTOMIZED_SERVICE_KEY, "")

    return (dst_bootstrap_dir, dst_model_serving_dir, relative_entry_fedml_format, enable_serverless_container,
            customized_image_entry_cmd, customized_readiness_check, customized_liveliness_check, customized_uri)


def handle_env_vars(environment, relative_entry_fedml_format, extra_envs, dst_bootstrap_dir, end_point_id, edge_id,
                    replica_rank, request_json):
    enable_custom_image = False if relative_entry_fedml_format != "" else True
    if not enable_custom_image:
        # For some image, the default user is root. Unified to fedml.
        environment["HOME"] = "/home/fedml"

    if request_json and ServerConstants.USER_ENCRYPTED_API_KEY in request_json:
        environment[ClientConstants.ENV_USER_ENCRYPTED_API_KEY] = request_json[ServerConstants.USER_ENCRYPTED_API_KEY]

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


def check_container_readiness(inference_http_port, infer_host="127.0.0.1", request_input_example=None,
                              readiness_check=ClientConstants.READINESS_PROBE_DEFAULT,
                              customized_uri=None):
    response_from_client_container = is_client_inference_container_ready(
        infer_host, inference_http_port, readiness_check=readiness_check,
        request_input_example=request_input_example, customized_uri=customized_uri)

    return response_from_client_container


def run_http_inference_with_curl_request(inference_url, inference_input_list, inference_output_list,
                                         inference_type="default", engine_type="default", timeout=None):
    return FedMLHttpInference.run_http_inference_with_curl_request(
        inference_url, inference_input_list, inference_output_list,
        inference_type=inference_type, engine_type=engine_type, timeout=timeout)


if __name__ == "__main__":
    pass
