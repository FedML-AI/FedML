import logging
import os
import pickle
import platform
import shutil
import time
import traceback
import yaml

import requests
import torch
import torch.nn
import tritonclient.http as http_client

import collections.abc

from fedml.computing.scheduler.comm_utils import sys_utils, security_utils
from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
import io

import docker
from ..scheduler_core.compute_cache_manager import ComputeCacheManager


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def start_deployment(end_point_id, end_point_name, model_id, model_version,
                     model_storage_local_path, model_bin_file, inference_model_name, inference_engine,
                     inference_http_port, inference_grpc_port, inference_metric_port,
                     inference_use_gpu, inference_memory_size,
                     inference_convertor_image, inference_server_image,
                     infer_host, model_is_from_open, model_params,
                     model_from_open, token, master_ip, edge_id):
    logging.info("Model deployment is starting...")

    use_simulation_test_without_triton = False
    model_metadata = {'name': inference_model_name,
                      'versions': ['1'], 'platform': 'onnxruntime_onnx',
                      'inputs': [{'name': 'input2', 'datatype': 'INT32', 'shape': [1, 24]},
                                 {'name': 'input1', 'datatype': 'FP32', 'shape': [1, 2]}],
                      'outputs': [{'name': 'output', 'datatype': 'FP32', 'shape': [1]}]}
    model_config = {
        "platform": "onnxruntime",
        "max_batch_size": 1,
        "input_size": [[1, 24], [1, 2]],
        "input_types": ["int", "float"],
        "input": [
            {
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": []
            }
        ],
        "output": [
            {
                "name": "output",
                "data_type": "TYPE_FP32",
                "dims": []
            }
        ]
    }

    num_gpus = 1
    try:
        ComputeCacheManager.get_instance().set_redis_params()
        with ComputeCacheManager.get_instance().get_redis_connection().lock(
                ComputeCacheManager.get_instance().get_device_run_lock_key(edge_id, end_point_id)
        ):
            num_gpus = ComputeCacheManager.get_instance().get_device_run_num_gpus(edge_id, end_point_id)
            num_gpus = int(num_gpus) if num_gpus is not None and str(num_gpus) != "" else 1
            gpu_ids = ComputeCacheManager.get_instance().get_device_run_gpu_ids(edge_id, end_point_id)
            if gpu_ids is not None:
                logging.info(f"cuda visible gpu ids: {gpu_ids}")
                gpu_list = JobRunnerUtils.trim_unavailable_gpu_ids(gpu_ids)
                logging.info(f"trimmed gpu ids {gpu_list}, num gpus {num_gpus}")
                if len(gpu_list) != num_gpus:
                    _, matched_gpu_num, matched_gpu_ids = JobRunnerUtils.request_gpu_ids(
                        num_gpus, JobRunnerUtils.get_realtime_gpu_available_ids())
                    gpu_ids = list(matched_gpu_ids)
                else:
                    gpu_ids = gpu_list
    except Exception as e:
        logging.info(f"Execption when fetching gpu ids. {traceback.format_exc()}")
        gpu_ids = None
        pass

    if not torch.cuda.is_available():
        gpu_attach_cmd = ""
    else:
        gpu_attach_cmd = "--gpus 1"
        if gpu_ids is not None and str(gpu_ids).strip() != "":
            gpu_id_map = map(lambda x: str(x), gpu_ids)
            gpu_ids_str = ','.join(gpu_id_map)
            gpu_attach_cmd = f"--gpus '\"device={gpu_ids_str}\"'"
        elif num_gpus is not None and str(num_gpus).strip() != "" and num_gpus > 0:
            gpu_attach_cmd = f"--gpus {num_gpus}"
        else:
            num_gpus = 1

    logging.info("Update docker environments...")

    sudo_prefix = "sudo "
    sys_name = platform.system()
    if sys_name == "Darwin":
        sudo_prefix = ""
        gpu_attach_cmd = ""

    running_model_name = ClientConstants.get_running_model_name(end_point_name,
                                                                inference_model_name,
                                                                model_version, end_point_id, model_id)

    # Check whether triton server is running.
    triton_server_is_running = False
    if not use_simulation_test_without_triton:
        triton_server_container_name = "{}".format(ClientConstants.FEDML_TRITON_SERVER_CONTAINER_NAME_PREFIX)
        if not ClientConstants.is_running_on_k8s():
            check_triton_server_running_cmds = "{}docker ps |grep {}".format(sudo_prefix, triton_server_container_name)
            running_process = ClientConstants.exec_console_with_script(check_triton_server_running_cmds,
                                                                       should_capture_stdout=True,
                                                                       should_capture_stderr=True)
            ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(running_process)
            if out is not None:
                out_str = sys_utils.decode_our_err_result(out)
                if str(out_str) != "":
                    triton_server_is_running = True

        logging.info("install nvidia docker...")

        # Setup nvidia docker related packages.
        if not ClientConstants.is_running_on_k8s():
            if sys_name == "Linux":
                if not triton_server_is_running:
                    os.system(sudo_prefix + "apt-get update")
                    os.system(
                        sudo_prefix + "apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin")
                    os.system("distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
                  && sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg;curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
                  && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
                     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
                    os.system(sudo_prefix + "apt-get update")
                    os.system(sudo_prefix + "apt-get install -y nvidia-docker2")
                    os.system(sudo_prefix + "systemctl restart docker")

    # Convert models from pytorch to onnx format
    if model_is_from_open:
        if model_from_open is None:
            return running_model_name, "", model_version, {}, {}

        logging.info("model binary file: {}".format(model_bin_file))
        with open(model_bin_file, 'rb') as model_pkl_file:
            if not torch.cuda.is_available():
                try:
                    open_model_params = CPUUnpickler(model_pkl_file).load()
                except Exception as ex:
                    logging.info("load model exceptions when using CPU_Unpickler: {}".format(traceback.format_exc()))
                    return "", "", model_version, model_metadata, model_config
            else:
                open_model_params = pickle.load(model_pkl_file)
            model_from_open.load_state_dict(open_model_params)
            model_from_open.eval()

        if inference_engine == ClientConstants.INFERENCE_ENGINE_TYPE_INT_TRITON:
            logging.info("convert the onnx model when the mode is from FedML® Nexus AI Platform..")
            logging.info("Input size {}, input types {}".format(model_params["input_size"],
                                                                model_params["input_types"]))
            input_size = model_params["input_size"]
            input_types = model_params["input_types"]

            dummy_input_list = []
            for index, input_i in enumerate(input_size):
                if input_types[index] == "int":
                    this_input = torch.randint(0, 1, input_i).clone().detach()
                else:
                    this_input = torch.zeros(input_i).clone().detach()
                dummy_input_list.append(this_input)

            onnx_model_path = os.path.join(model_storage_local_path,
                                           ClientConstants.FEDML_CONVERTED_MODEL_DIR_NAME,
                                           running_model_name, ClientConstants.INFERENCE_MODEL_VERSION)
            if not os.path.exists(onnx_model_path):
                os.makedirs(onnx_model_path, exist_ok=True)
            onnx_model_path = os.path.join(onnx_model_path, "model.onnx")

            convert_model_to_onnx(model_from_open, onnx_model_path, dummy_input_list, input_size)
        elif ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEFAULT:  # we do not convert the model to onnx in llm
            logging.info("LLM model loaded from the open")
        else:
            raise Exception("Unsupported inference engine type: {}".format(inference_engine))
    elif model_is_from_open == False or model_is_from_open is None:
        model_location = os.path.join(model_storage_local_path, "fedml_model.bin")
        try:
            model = torch.jit.load(model_location)
            model.eval()
        except Exception as e:
            logging.info(
                "Cannot locate the .bin file, will read it from"
                " the fedml_model_config.yaml with the key [local_model_dir] ")
            model_config_path = os.path.join(model_storage_local_path, "fedml_model_config.yaml")
            with open(model_config_path, 'r') as file:
                config = yaml.safe_load(file)
                # Resource related
                use_gpu = config.get('use_gpu', False)
                usr_indicated_wait_time = config.get('deploy_timeout', 100)
                usr_indicated_retry_cnt = max(int(usr_indicated_wait_time) // 10, 1)
                inference_image_name = config.get('inference_image_name',
                                                  ClientConstants.INFERENCE_SERVER_CUSTOME_IMAGE)
                # Source code dir, bootstrap dir, data cache dir
                src_code_dir = os.path.join(model_storage_local_path, config.get('source_code_dir', ""))

                # Get the bootstrap commands inside the yaml file
                bootstrap_cmds_str_frm_yaml = config.get('bootstrap', "")
                if bootstrap_cmds_str_frm_yaml != "":
                    auto_gen_bootstrap_file_name = "fedml-deploy-bootstrap-auto-gen.sh"
                    src_bootstrap_file_path = os.path.join(model_storage_local_path, auto_gen_bootstrap_file_name)
                    with open(src_bootstrap_file_path, 'w') as f:
                        f.write(bootstrap_cmds_str_frm_yaml)
                else:
                    src_bootstrap_file_path = ""

                data_cache_dir_input = config.get('data_cache_dir', "")
                request_input_example = config.get('request_input_example', None)
                extra_envs = config.get('environment_variables', None)
                logging.info(
                    f"src_code_dir: {src_code_dir}, bootstrap_src_path: {src_bootstrap_file_path},"
                    f" data_cache_dir_input: {data_cache_dir_input}")
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
                # Serving dir inside docker
                dst_model_serving_dir = "/home/fedml/models_serving"
                relative_entry = config.get('entry_point')
                if src_bootstrap_file_path != "":
                    dst_bootstrap_dir = os.path.join(dst_model_serving_dir, auto_gen_bootstrap_file_name)
                else:
                    dst_bootstrap_dir = ""

            if src_code_dir == "":
                raise Exception("Please indicate source_code_dir in the fedml_model_config.yaml")
            if relative_entry == "":
                raise Exception("Please indicate main_entry in the fedml_model_config.yaml")

        if inference_engine == ClientConstants.INFERENCE_ENGINE_TYPE_INT_TRITON:
            # configuration passed by user in the Cli
            input_size = model_params["input_size"]
            input_types = model_params["input_types"]
            logging.info("convert the onnx model when the mode is from the general PyTorch...")
            logging.info("Input size {}, input types {}".format(model_params["input_size"],
                                                                model_params["input_types"]))
            dummy_input_list = []
            for index, input_i in enumerate(input_size):
                if input_types[index] == "int":
                    this_input = torch.randint(0, 1, input_i).clone().detach()
                else:
                    this_input = torch.zeros(input_i).clone().detach()
                dummy_input_list.append(this_input)

            onnx_model_path = os.path.join(model_storage_local_path,
                                           ClientConstants.FEDML_CONVERTED_MODEL_DIR_NAME,
                                           running_model_name, ClientConstants.INFERENCE_MODEL_VERSION)
            logging.info("converted onnx model path: {}".format(onnx_model_path))
            if not os.path.exists(onnx_model_path):
                os.makedirs(onnx_model_path, exist_ok=True)
            onnx_model_path = os.path.join(onnx_model_path, "model.onnx")

            convert_model_to_onnx(model, onnx_model_path, dummy_input_list, input_size)

    logging.info("move converted model to serving dir for inference...")
    model_serving_dir = ClientConstants.get_model_serving_dir()
    if not os.path.exists(model_serving_dir):
        os.makedirs(model_serving_dir, exist_ok=True)
    converted_model_path = os.path.join(model_storage_local_path, ClientConstants.FEDML_CONVERTED_MODEL_DIR_NAME)
    if os.path.exists(converted_model_path):
        model_file_list = os.listdir(converted_model_path)
        for model_file in model_file_list:
            src_model_file = os.path.join(converted_model_path, model_file)
            dst_model_file = os.path.join(model_serving_dir, model_file)
            if os.path.isdir(src_model_file):
                if not os.path.exists(dst_model_file):
                    shutil.copytree(src_model_file, dst_model_file, copy_function=shutil.copy,
                                    ignore_dangling_symlinks=True)
            else:
                if not os.path.exists(dst_model_file):
                    shutil.copyfile(src_model_file, dst_model_file)

    if inference_engine == ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEFAULT:
        logging.info(f"master ip: {master_ip}, worker ip: {infer_host}")
        if infer_host == master_ip:
            logging.info("infer_host is the same as master ip, will use 127.0.0.1 to avoid firewall issue")
            infer_host = "127.0.0.1"

        try:
            client = docker.from_env()
        except Exception:
            logging.error("Failed to connect to the docker daemon, please ensure that you have "
                          "installed Docker Desktop or Docker Engine, and the docker is running")
            return "", "", None, None, None

        default_server_container_name = "{}".format(
            ClientConstants.FEDML_DEFAULT_SERVER_CONTAINER_NAME_PREFIX) + "__" +\
            security_utils.get_content_hash(running_model_name)

        try:
            exist_container_obj = client.containers.get(default_server_container_name)
        except docker.errors.NotFound:
            exist_container_obj = None
        except docker.errors.APIError:
            raise Exception("Failed to get the container object")

        if exist_container_obj is not None:
            client.api.remove_container(exist_container_obj.id, v=True, force=True)
        device_requests = []
        if use_gpu:
            if gpu_ids is not None:
                gpu_id_list = map(lambda x: str(x), gpu_ids)
                device_requests.append(
                    docker.types.DeviceRequest(device_ids=list(gpu_id_list), capabilities=[['gpu']]))
            else:
                device_requests.append(
                    docker.types.DeviceRequest(count=num_gpus, capabilities=[['gpu']]))
        logging.info("Start pulling the inference image..., may take a few minutes...")
        # TODO:only pull if the image is not in the local
        client.images.pull(inference_image_name)
        logging.info("Start creating the inference container...")

        volumns = []
        binds = {}
        environment = {}

        # Optional
        if src_data_cache_dir != "":
            volumns.append(src_data_cache_dir)
            binds[src_data_cache_dir] = {
                "bind": dst_data_cache_dir,
                "mode": "rw"
            }
            environment["DATA_CACHE_FOLDER"] = dst_data_cache_dir

        # Default
        volumns.append(src_code_dir)
        binds[src_code_dir] = {
            "bind": dst_model_serving_dir,
            "mode": "rw"
        }
        environment["BOOTSTRAP_DIR"] = dst_bootstrap_dir
        environment["MAIN_ENTRY"] = relative_entry

        if extra_envs is not None:
            for key in extra_envs:
                environment[key] = extra_envs[key]

        new_container = client.api.create_container(
            image=inference_image_name,
            name=default_server_container_name,
            volumes=volumns,
            ports=[2345],  # port open inside the container
            # entrypoint=["python3", relative_entry],
            environment=environment,
            host_config=client.api.create_host_config(
                binds=binds,
                port_bindings={
                    2345: None  # randomly open a port on the host
                },
                device_requests=device_requests,
                # mem_limit = "8g",   # Could also be configured in the docker desktop setting
            ),
            detach=True,
        )
        client.api.start(container=new_container.get("Id"))

        # Get the port allocation
        cnt = 0
        while True:
            cnt += 1
            try:  # check dynamic port allocation
                port_info = client.api.port(new_container.get("Id"), 2345)
                inference_http_port = port_info[0]["HostPort"]
                logging.info("inference_http_port: {}".format(inference_http_port))
                break
            except:
                if cnt >= 5:
                    raise Exception("Failed to get the port allocation")
                time.sleep(3)

        # Logging the info from the container
        log_deployment_result(end_point_id, model_id, default_server_container_name,
                              ClientConstants.CMD_TYPE_RUN_DEFAULT_SERVER,
                              running_model_name, inference_engine, inference_http_port, inference_type="default",
                              retry_interval=10, deploy_attempt_threshold=usr_indicated_retry_cnt,
                              request_input_example=request_input_example, infer_host=infer_host)

        # Check if the inference server is ready
        inference_output_url, running_model_version, ret_model_metadata, ret_model_config = \
            get_model_info(running_model_name, inference_engine, inference_http_port,
                           infer_host, inference_type="default", request_input_example=request_input_example)

        if inference_output_url == "":
            return running_model_name, "", None, None, None

        # testing the inference container
        test_input = ret_model_metadata["inputs"]

        try:
            inference_response = run_http_inference_with_curl_request(inference_output_url, test_input, [],
                                                                      inference_type="default")
            logging.info(f"Tested the inference backend with {test_input}, the response is {inference_response}")
        except Exception as e:
            logging.info("Tested the inference backend, exceptions occurred: {}".format(traceback.format_exc()))
            inference_output_url = ""

        model_metadata = ret_model_metadata
        logging.info(model_metadata)
    elif inference_engine == ClientConstants.INFERENCE_ENGINE_TYPE_INT_TRITON:
        logging.info("prepare to run triton server...")
        if not use_simulation_test_without_triton:
            if not triton_server_is_running and not ClientConstants.is_running_on_k8s():
                triton_server_cmd = "{}docker stop {}; {}docker rm {}; {}docker run --name {} {} -p{}:8000 " \
                                    "-p{}:8001 -p{}:8002 " \
                                    "--shm-size {} " \
                                    "-v {}:/models {} " \
                                    "bash -c \"pip install transformers && tritonserver --strict-model-config=false " \
                                    "--model-control-mode=poll --repository-poll-secs={} " \
                                    "--model-repository=/models\" ".format(sudo_prefix, triton_server_container_name,
                                                                           sudo_prefix, triton_server_container_name,
                                                                           sudo_prefix, triton_server_container_name,
                                                                           gpu_attach_cmd,
                                                                           inference_http_port,
                                                                           inference_grpc_port,
                                                                           inference_metric_port,
                                                                           inference_memory_size,
                                                                           model_serving_dir,
                                                                           inference_server_image,
                                                                           ClientConstants.FEDML_MODEL_SERVING_REPO_SCAN_INTERVAL)
                logging.info("Run triton inference server: {}".format(triton_server_cmd))
                triton_server_process = ClientConstants.exec_console_with_script(triton_server_cmd,
                                                                                 should_capture_stdout=False,
                                                                                 should_capture_stderr=False,
                                                                                 no_sys_out_err=True)
                log_deployment_result(end_point_id, model_id, triton_server_container_name,
                                      ClientConstants.CMD_TYPE_RUN_TRITON_SERVER, triton_server_process.pid,
                                      running_model_name, inference_engine, inference_http_port)

            inference_output_url, running_model_version, ret_model_metadata, ret_model_config = \
                get_model_info(running_model_name, inference_engine, inference_http_port, infer_host)
            if inference_output_url != "":
                # Send the test request to the inference backend and check if the response is normal
                input_json, output_json = build_inference_req(end_point_name, inference_model_name,
                                                              token, ret_model_metadata)
                try:
                    inference_response = run_http_inference_with_curl_request(inference_output_url,
                                                                              input_json["inputs"],
                                                                              input_json["outputs"])
                    logging.info("Tested the inference backend, the response is {}".format(inference_response))
                except Exception as e:
                    logging.info("Tested the inference backend, exceptions occurred: {}".format(traceback.format_exc()))
                    inference_output_url = ""

                if inference_output_url != "":
                    logging.info(
                        "Deploy model successfully, inference url: {}, model metadata: {}, model config: {}".format(
                            inference_output_url, model_metadata, model_config))
                    model_metadata = ret_model_metadata
                    model_config = ret_model_config
        else:
            inference_output_url = f"http://localhost:{inference_http_port}/v2/models/{running_model_name}/versions/1/infer"
    else:
        raise Exception("inference engine {} is not supported".format(inference_engine))

    return running_model_name, inference_output_url, model_version, model_metadata, model_config


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
                     inference_type="default", request_input_example=None, infer_host="127.0.0.1"):
    sudo_prefix = "sudo "
    sys_name = platform.system()
    if sys_name == "Darwin":
        sudo_prefix = ""

    if cmd_type == ClientConstants.CMD_TYPE_CONVERT_MODEL:
        convert_model_container_name = "{}_{}_{}".format(ClientConstants.FEDML_CONVERT_MODEL_CONTAINER_NAME_PREFIX,
                                                         str(end_point_id),
                                                         str(model_id))
        docker_ps_cmd = "{}docker ps -a;exit".format(sudo_prefix, convert_model_container_name)
        docker_ps_process = ClientConstants.exec_console_with_script(docker_ps_cmd,
                                                                     should_capture_stdout=True,
                                                                     should_capture_stderr=True)
        ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(docker_ps_process)
        if out is not None:
            out_str = sys_utils.decode_our_err_result(out)
            if str(out_str).find(convert_model_container_name) == -1:
                return True
            else:
                return False
        else:
            return True
    elif cmd_type == ClientConstants.CMD_TYPE_RUN_TRITON_SERVER:
        try:
            inference_output_url, model_version, model_metadata, model_config = \
                get_model_info(model_name, inference_engine, inference_port, inference_type=inference_type)
            logging.info("Log test for deploying model successfully, inference url: {}, "
                         "model metadata: {}, model config: {}".
                         format(inference_output_url, model_metadata, model_config))
            if inference_output_url != "":
                return True
        except Exception as e:
            pass
        return False
    elif cmd_type == ClientConstants.CMD_TYPE_RUN_DEFAULT_SERVER:
        # TODO: Exited Quickly if the container is Exited or Removed
        # If the container has exited, return True, means we should exit the logs
        # container_name = "{}".format(ClientConstants.FEDML_DEFAULT_SERVER_CONTAINER_NAME_PREFIX) + "__" + \
        #                             security_utils.get_content_hash(model_name)
        try:
            inference_output_url, model_version, model_metadata, model_config = \
                get_model_info(model_name, inference_engine, inference_port, infer_host,
                               inference_type=inference_type, request_input_example=request_input_example)
            logging.info("Log test for deploying model successfully, inference url: {}, "
                         "model metadata: {}, model config: {}".
                         format(inference_output_url, model_metadata, model_config))
            if inference_output_url != "":
                return True
        except Exception as e:
            pass
        return False


def log_deployment_result(end_point_id, model_id, cmd_container_name, cmd_type,
                          inference_model_name, inference_engine,
                          inference_http_port, inference_type="default",
                          retry_interval=10, deploy_attempt_threshold=10,
                          request_input_example=None, infer_host="127.0.0.1"):
    deploy_attempt = 0
    last_out_logs = ""
    last_err_logs = ""

    while True:
        if not ClientConstants.is_running_on_k8s():
            logging.info(f"Test: {inference_http_port}, Attempt: {deploy_attempt} / {deploy_attempt_threshold}")

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
                out_logs = container_obj.logs(stdout=True, stderr=False, stream=False, follow=False)
                err_logs = container_obj.logs(stdout=False, stderr=True, stream=False, follow=False)
                if err_logs is not None:
                    err_logs = sys_utils.decode_our_err_result(err_logs)
                    added_logs = str(err_logs).replace(last_err_logs, "")
                    if len(added_logs) > 0:
                        log_str = f"logs from docker: {format(added_logs)}"
                        if added_logs.startswith("ERROR:") or added_logs.startswith("CRITICAL:"):
                            logging.error(log_str)
                            last_err_logs = err_logs
                        elif added_logs.startswith("WARNING:"):
                            logging.warning(log_str)
                            last_out_logs = err_logs
                        elif added_logs.startswith("DEBUG:"):
                            logging.debug(log_str)
                            last_out_logs = err_logs
                        else:
                            logging.info(log_str)
                            last_out_logs = err_logs

                if out_logs is not None:
                    out_logs = sys_utils.decode_our_err_result(out_logs)
                    added_logs = str(out_logs).replace(last_out_logs, "")
                    if len(added_logs) > 0:
                        logging.info(f"Logs from docker: {format(added_logs)}")
                    last_out_logs = out_logs

                if container_obj.status == "exited":
                    logging.info("Container {} has exited, automatically"
                                 " remove it ...".format(cmd_container_name))
                    client.api.remove_container(container_obj.id, v=True, force=True)
                    break

        # should_exit_logs will ping the inference container
        # return True if ready
        if should_exit_logs(end_point_id, model_id, cmd_type, inference_model_name, inference_engine,
                            inference_http_port, inference_type, request_input_example,
                            infer_host):
            break

        # Not yet ready, retry
        deploy_attempt += 1
        if deploy_attempt >= deploy_attempt_threshold:
            logging.info(f"Model {inference_model_name} deploy reached max attempt {deploy_attempt_threshold}, "
                         "exiting the deployment...")
            break

        logging.info(f"Model {inference_model_name} not yet ready, retry in {retry_interval} seconds...")
        time.sleep(retry_interval)


def is_client_inference_container_ready(infer_url_host, inference_http_port, inference_model_name, local_infer_url,
                                        inference_type="default", model_version="", request_input_example=None):
    logging.info(f"Inference type: {inference_type}, infer_url_host {infer_url_host}")

    if inference_type == "default":
        default_client_container_ready_url = "http://{}:{}/ready".format(infer_url_host, inference_http_port)
        response = None
        try:
            response = requests.get(default_client_container_ready_url)
        except:
            pass
        if not response or response.status_code != 200:
            return "", "", {}, {}

        # Report the deployed model info
        model_metadata = {}
        if request_input_example is not None and len(request_input_example) > 0:
            model_metadata["inputs"] = request_input_example
        else:
            model_metadata["inputs"] = {"text": "What is a good cure for hiccups?"}
        model_metadata["outputs"] = []
        model_metadata["type"] = "default"
        return "http://{}:{}/predict".format(infer_url_host, inference_http_port), None, model_metadata, None
    else:
        triton_client = http_client.InferenceServerClient(url=local_infer_url, verbose=False)
        wait_count = 0
        while True:
            if not triton_client.is_model_ready(
                    model_name=inference_model_name, model_version=model_version
            ):
                logging.info(f"model {inference_model_name} not yet ready")
                time.sleep(1)
                wait_count += 1
                if wait_count >= 15:
                    return "", model_version, {}, {}
            else:
                break

        model_metadata = triton_client.get_model_metadata(model_name=inference_model_name, model_version=model_version)
        model_config = triton_client.get_model_config(model_name=inference_model_name, model_version=model_version)
        version_list = model_metadata.get("versions", None)
        if version_list is not None and len(version_list) > 0:
            model_version = version_list[0]
        else:
            model_version = ClientConstants.INFERENCE_MODEL_VERSION

        inference_output_url = "http://{}:{}/{}/models/{}/versions/{}/infer".format(infer_url_host,
                                                                                    inference_http_port,
                                                                                    ClientConstants.INFERENCE_INFERENCE_SERVER_VERSION,
                                                                                    inference_model_name,
                                                                                    model_version)

        return inference_output_url, model_version, model_metadata, model_config


def get_model_info(model_name, inference_engine, inference_http_port, infer_host="127.0.0.1", is_hg_model=False,
                   inference_type="default", request_input_example=None):
    if model_name is None:
        return "", "", {}, {}

    local_infer_url = "{}:{}".format(infer_host, inference_http_port)
    logging.info(f"The infer_url_host is {infer_host}")
    logging.info(f"Local infer url: {local_infer_url}.")

    if is_hg_model:
        inference_model_name = "{}_{}_inference".format(model_name, str(inference_engine))
    else:
        inference_model_name = model_name

    response_from_client_container = is_client_inference_container_ready(
        infer_host, inference_http_port, inference_model_name, local_infer_url,
        inference_type, model_version="", request_input_example=request_input_example)

    logging.info(f"The res is {response_from_client_container}")
    return response_from_client_container


def run_http_inference_with_curl_request(inference_url, inference_input_list, inference_output_list,
                                         inference_type="default"):
    model_inference_result = {}
    model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
    print("inference_url: {}".format(inference_url))
    print("inference_input_list: {}".format(inference_input_list))
    if inference_type == "default":
        model_inference_json = inference_input_list
    else:  # triton
        model_inference_json = {
            "inputs": inference_input_list,
            "outputs": inference_output_list
        }

    try:
        response = requests.post(inference_url, headers=model_api_headers, json=model_inference_json)
        if response.status_code == 200:
            model_inference_result = response.json()
    except Exception as e:
        print("Error in running inference: {}".format(e))

    return model_inference_result


def convert_model_to_onnx(
        torch_model, output_path: str, dummy_input_list, input_size: int, input_is_tensor=True
) -> None:
    from collections import OrderedDict
    import torch
    from torch.onnx import TrainingMode

    torch.onnx.export(torch_model,  # model being run
                      dummy_input_list if input_is_tensor else tuple(dummy_input_list),
                      # model input (or a tuple for multiple inputs)
                      f=output_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=["input1", "input2"],
                      # the model's input names
                      output_names=['output'],  # the model's output names
                      training=TrainingMode.EVAL,
                      verbose=True,
                      dynamic_axes={"input1": {0: "batch_size"},
                                    "input2": {0: "batch_size"},
                                    "output": {0: "batch_size"}}
                      )


def test_start_triton_server(model_serving_dir):
    sudo_prefix = "sudo "
    sys_name = platform.system()
    if sys_name == "Darwin":
        sudo_prefix = ""
        gpu_attach_cmd = ""

    triton_server_container_name = "{}".format(ClientConstants.FEDML_TRITON_SERVER_CONTAINER_NAME_PREFIX)
    triton_server_cmd = "{}docker stop {}; {}docker rm {}; {}docker run --name {} {} -p{}:8000 " \
                        "-p{}:8001 -p{}:8002 " \
                        "--shm-size {} " \
                        "-v {}:/models {} " \
                        "bash -c \"pip install transformers && tritonserver --strict-model-config=false " \
                        "--model-control-mode=poll --repository-poll-secs={} " \
                        "--model-repository=/models\" ".format(sudo_prefix, triton_server_container_name,
                                                               sudo_prefix, triton_server_container_name,
                                                               sudo_prefix, triton_server_container_name,
                                                               gpu_attach_cmd,
                                                               ClientConstants.INFERENCE_HTTP_PORT,
                                                               ClientConstants.INFERENCE_GRPC_PORT,
                                                               8002,
                                                               "4096m",
                                                               model_serving_dir,
                                                               ClientConstants.INFERENCE_SERVER_IMAGE,
                                                               ClientConstants.FEDML_MODEL_SERVING_REPO_SCAN_INTERVAL)
    logging.info("Run triton inference server: {}".format(triton_server_cmd))
    triton_server_process = ClientConstants.exec_console_with_script(triton_server_cmd,
                                                                     should_capture_stdout=False,
                                                                     should_capture_stderr=False,
                                                                     no_sys_out_err=True)


def test_convert_pytorch_model_to_onnx(model_net_file, model_bin_file, model_name, model_in_params):
    torch_model = torch.jit.load(model_net_file)
    with open(model_bin_file, 'rb') as model_pkl_file:
        model_state_dict = pickle.load(model_pkl_file)
        torch_model.load_state_dict(model_state_dict)
        torch_model.eval()

    input_size = model_in_params["input_size"]
    input_types = model_in_params["input_types"]

    dummy_input_list = []
    for index, input_i in enumerate(input_size):
        if input_types[index] == "int":
            this_input = torch.tensor(torch.randint(0, 1, input_i))
        else:
            this_input = torch.tensor(torch.zeros(input_i))
        dummy_input_list.append(this_input)

    onnx_model_dir = os.path.join(ClientConstants.get_model_cache_dir(),
                                  ClientConstants.FEDML_CONVERTED_MODEL_DIR_NAME,
                                  model_name, ClientConstants.INFERENCE_MODEL_VERSION)
    if not os.path.exists(onnx_model_dir):
        os.makedirs(onnx_model_dir, exist_ok=True)
    onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")

    convert_model_to_onnx(torch_model, onnx_model_path, dummy_input_list, input_size,
                          input_is_tensor=True)

    model_serving_dir = os.path.join(ClientConstants.get_model_cache_dir(),
                                     ClientConstants.FEDML_CONVERTED_MODEL_DIR_NAME)
    return model_serving_dir


def start_gpu_model_load_process():
    from multiprocessing import Process
    import time
    process = Process(target=load_gpu_model_to_cpu_device)
    process.start()
    while True:
        time.sleep(1)


def load_gpu_model_to_cpu_device():
    import pickle
    import io
    import torch

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)

    model_file = "/home/fedml/.fedml/fedml-client/fedml/models/theta_rec_auc_81_single_label/theta_rec_auc_81_single_label"
    with open(model_file, "rb") as model_pkl_file:
        if not torch.cuda.is_available():
            model = CPU_Unpickler(model_pkl_file).load()
            if model is None:
                print("Failed to load gpu model to cpu device")
            else:
                print("Succeeded to load gpu model to cpu device")


if __name__ == "__main__":
    start_gpu_model_load_process()

    model_serving_dir = test_convert_pytorch_model_to_onnx("./sample-open-training-model-net",
                                                           "./sample-open-training-model",
                                                           "rec-model",
                                                           {"input_size": [[1, 24], [1, 2]],
                                                            "input_types": ["int", "float"]})

    test_start_triton_server(model_serving_dir)

    # input_data = {"model_version": "v0-Sun Feb 05 12:17:16 GMT 2023",
    #               "model_name": "model_414_45_open-model-test_v0-Sun-Feb-05-12-17-16-GMT-2023",
    #               # "data": "file:///Users/alexliang/fedml_data/mnist-image.png",
    #               "data": "https://raw.githubusercontent.com/niyazed/triton-mnist-example/master/images/sample_image.png",
    #               "end_point_id": 414, "model_id": 45, "token": "a09a18a14c4c4d89a8d5f9515704c073"}
    #
    # data_list = list()
    # data_list.append(input_data["data"])
    # run_http_inference_with_lib_http_api_with_image_data(input_data["model_name"],
    #                                                      5001, 1, data_list, "")
    #
    #
    # class LogisticRegression(torch.nn.Module):
    #     def __init__(self, input_dim, output_dim):
    #         super(LogisticRegression, self).__init__()
    #         self.linear = torch.nn.Linear(input_dim, output_dim)
    #
    #     def forward(self, x):
    #         outputs = torch.sigmoid(self.linear(x))
    #         return outputs
    #
    #
    # model = LogisticRegression(28 * 28, 10)
    # checkpoint = {'model': model}
    # model_net_file = "/Users/alexliang/fedml-client/fedml/models/open-model-test/model-net.pt"
    # torch.save(checkpoint, model_net_file)
    #
    # with open("/Users/alexliang/fedml-client/fedml/models/open-model-test/open-model-test", 'rb') as model_pkl_file:
    #     model_params = pickle.load(model_pkl_file)
    #     # torch.save(model_params, "/Users/alexliang/fedml-client/fedml/models/open-model-test/a.pt")
    #     # model = torch.load("/Users/alexliang/fedml-client/fedml/models/open-model-test/a.pt")
    #     loaded_checkpoint = torch.load(model_net_file)
    #     loaded_model = loaded_checkpoint["model"]
    #     loaded_model.load_state_dict(model_params)
    #     for parameter in loaded_model.parameters():
    #         parameter.requires_grad = False
    #     loaded_model.eval()
    #     input_names = {"x": 0}
    #     convert_model_to_onnx(loaded_model, "/Users/alexliang/fedml-client/fedml/models/open-model-test/a.onnx",
    #                           input_names, 28 * 28)

    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--cf", "-c", help="config file")
    # parser.add_argument("--role", "-r", type=str, default="client", help="role")
    # parser.add_argument("--model_storage_local_path", "-url", type=str, default="/home/ubuntu",
    #                     help="model storage local path")
    # parser.add_argument("--inference_model_name", "-n", type=str, default="fedml-model",
    #                     help="inference model name")
    # parser.add_argument("--inference_engine", "-engine", type=str, default="ONNX", help="inference engine")
    # parser.add_argument("--inference_http_port", "-http", type=int, default=8000, help="inference http port")
    # parser.add_argument("--inference_grpc_port", "-gprc", type=int, default=8001, help="inference grpc port")
    # parser.add_argument("--inference_metric_port", "-metric", type=int, default=8002, help="inference metric port")
    # parser.add_argument("--inference_use_gpu", "-gpu", type=str, default="gpu", help="inference use gpu")
    # parser.add_argument("--inference_memory_size", "-mem", type=str, default="256m", help="inference memory size")
    # parser.add_argument("--inference_convertor_image", "-convertor", type=str,
    #                     default=ClientConstants.INFERENCE_CONVERTOR_IMAGE, help="inference convertor image")
    # parser.add_argument("--inference_server_image", "-server", type=str,
    #                     default=ClientConstants.INFERENCE_SERVER_IMAGE, help="inference server image")
    # args = parser.parse_args()
    # args.user = args.user
    #
    # pip_source_dir = os.path.dirname(__file__)
    # __running_model_name, __inference_output_url, __model_version, __model_metadata, __model_config = \
    #     start_deployment(
    #         args.model_storage_local_path,
    #         args.inference_model_name,
    #         args.inference_engine,
    #         args.inference_http_port,
    #         args.inference_grpc_port,
    #         args.inference_metric_port,
    #         args.inference_use_gpu,
    #         args.inference_memory_size,
    #         args.inference_convertor_image,
    #         args.inference_server_image)
    # print("Model deployment results, running model name: {}, url: {}, model metadata: {}, model config: {}".format(
    #     __running_model_name, __inference_output_url, __model_metadata, __model_config))
