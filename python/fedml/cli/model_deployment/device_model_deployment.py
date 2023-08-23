import logging
import os
import pickle
import platform
import shutil
import time
import traceback

import requests
import torch
import torch.nn
import tritonclient.http as http_client

import collections.abc

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from fedml.cli.model_deployment.device_client_constants import ClientConstants
import io


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
                     model_from_open, token):
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

    if not torch.cuda.is_available():
        gpu_attach_cmd = ""
    else:
        gpu_attach_cmd = "--gpus all"

    logging.info("Update docker environments...")

    sudo_prefix = "sudo "
    sys_name = platform.system()
    if sys_name == "Darwin":
        sudo_prefix = ""
        gpu_attach_cmd = ""

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
                out_str = out.decode(encoding="utf-8")
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
        running_model_name = ClientConstants.get_running_model_name(end_point_name,
                                                                    inference_model_name,
                                                                    model_version, end_point_id, model_id)
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
            logging.info("convert the onnx model when the mode is from the MLOps platform...")
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
        elif ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEEPSPEED:  # we do not convert the model to onnx in llm
            logging.info("LLM model loaded from the open")
        else:
            raise Exception("Unsupported inference engine type: {}".format(inference_engine))
    elif model_is_from_open == False:
        running_model_name = ClientConstants.get_running_model_name(end_point_name,
                                                                    inference_model_name,
                                                                    model_version, end_point_id, model_id)
        model_location = os.path.join(model_storage_local_path, "fedml_model.bin")
        try:
            model = torch.jit.load(model_location)
            model.eval()
        except Exception as e:
            logging.info(
                "Cannot locate the .bin file, will read it from the fedml_model_cofig.yaml with the key [local_model_dir] ")
            import yaml
            local_model_location = os.path.join(model_storage_local_path, "fedml_model_config.yaml")

            with open(local_model_location, 'r') as file:
                config = yaml.safe_load(file)
                local_model_dir = config.get('local_model_dir', "")
                inference_image_name = config.get('inference_image_name', "")
            
            if local_model_dir == "":
                raise Exception("Please indicate local_model_dir in the fedml_model_config.yaml")
            if inference_image_name == "":
                raise Exception("Please indicate inference_image_name in the fedml_model_config.yaml")
            
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

    if inference_engine == ClientConstants.INFERENCE_ENGINE_TYPE_INT_DEEPSPEED:
        logging.info(f"local_model_dir: {local_model_dir}")
        logging.info(f"inference_image_name: {inference_image_name}")
        inference_server_image = inference_image_name
        inference_http_port = 2345  # TODO: using a threading pool to manage the model
        local_model_dir = local_model_dir
        llm_server_container_name = "{}".format(ClientConstants.FEDML_LLM_SERVER_CONTAINER_NAME_PREFIX)
        volume_dst_loc = "code/model_and_config"
        llm_server_cmd = "{}docker stop {}; {}docker rm {}; {}docker run --gpus all --name {} -p{}:2345 " \
                         "-v {}:/{} {}".format(sudo_prefix, llm_server_container_name,
                                               sudo_prefix, llm_server_container_name,
                                               sudo_prefix, llm_server_container_name,
                                               inference_http_port,
                                               local_model_dir,
                                               volume_dst_loc,
                                               inference_server_image)
        logging.info("Run llm inference server: {}".format(llm_server_cmd))
        llm_server = ClientConstants.exec_console_with_script(llm_server_cmd,
                                                              should_capture_stdout=False,
                                                              should_capture_stderr=False,
                                                              no_sys_out_err=True)
        # report the status
        log_deployment_result(end_point_id, model_id, llm_server_container_name,
                              ClientConstants.CMD_TYPE_RUN_TRITON_SERVER, llm_server.pid,
                              running_model_name, inference_engine, inference_http_port, inference_type="llm")
        inference_output_url, running_model_version, ret_model_metadata, ret_model_config = \
            get_model_info(running_model_name, inference_engine, inference_http_port, infer_host, inference_type="llm")

        # testing
        test_input = {"inputs": {"text": "What is a good cure for hiccups?"}}
        if inference_output_url != "":
            try:
                inference_response = run_http_inference_with_curl_request(inference_output_url, test_input, [],
                                                                          inference_type="llm")
                logging.info("Tested the inference backend, the response is {}".format(inference_response))
            except Exception as e:
                logging.info("Tested the inference backend, exceptions occurred: {}".format(traceback.format_exc()))
                inference_output_url = ""
        else:
            raise Exception("Failed to get the inference output url")
        model_metadata = ret_model_metadata

        # metadata to report
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


def should_exit_logs(end_point_id, model_id, cmd_type, cmd_process_id, model_name, inference_engine, inference_port,
                     inference_type=None):
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
            out_str = out.decode(encoding="utf-8")
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


def log_deployment_result(end_point_id, model_id, cmd_container_name, cmd_type,
                          cmd_process_id, inference_model_name, inference_engine,
                          inference_http_port, inference_type=None):
    sudo_prefix = "sudo "
    sys_name = platform.system()
    if sys_name == "Darwin":
        sudo_prefix = ""

    last_out_logs = ""
    last_err_logs = ""
    deployment_count = 0
    while True:
        if not ClientConstants.is_running_on_k8s():
            logs_cmd = "{}docker logs {}".format(sudo_prefix, cmd_container_name)
            logs_process = ClientConstants.exec_console_with_script(logs_cmd,
                                                                    should_capture_stdout=True,
                                                                    should_capture_stderr=True)
            ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(logs_process)
            if out is not None:
                out_str = out.decode(encoding="utf-8")
                added_logs = str(out_str).replace(last_out_logs, "")
                if len(added_logs) > 0:
                    logging.info("{}".format(added_logs))
                last_out_logs = out_str
            elif err is not None:
                err_str = err.decode(encoding="utf-8")
                added_logs = str(err_str).replace(last_err_logs, "")
                if len(added_logs) > 0:
                    logging.info("{}".format(added_logs))
                last_err_logs = err_str

        time.sleep(3)
        deployment_count += 1
        if deployment_count >= 5:
            break

        if should_exit_logs(end_point_id, model_id, cmd_type, cmd_process_id,
                            inference_model_name, inference_engine, inference_http_port, inference_type):
            break


def get_model_info(model_name, inference_engine, inference_http_port, infer_host=None, is_hg_model=False,
                   inference_type=None):
    local_ip = ClientConstants.get_local_ip()
    if infer_host is not None and infer_host != "127.0.0.1":
        infer_url_host = infer_host
    else:
        infer_url_host = local_ip
    logging.info(f"The infer_url_host is {infer_url_host}")
    if inference_type == "llm":
        llm_server_test_ready_url = "http://{}:{}/ready".format(infer_url_host, inference_http_port)
        wait_count = 0
        while True:
            response = None
            try:
                response = requests.get(llm_server_test_ready_url)
            except:
                pass
            if not response or response.status_code != 200:
                logging.info(f"model {model_name} not yet ready")
                time.sleep(10)
                wait_count += 1
                if wait_count >= 15:
                    raise Exception("Can not get response from {}".format(llm_server_test_ready_url))
            else:
                break
        model_metadata = {}
        model_metadata["inputs"] = {"text": "What is a good cure for hiccups?"}
        model_metadata["outputs"] = []
        model_metadata["type"] = "llm"
        return "http://{}:{}/predict".format(infer_url_host, inference_http_port), None, model_metadata, None
    local_infer_url = "{}:{}".format(infer_url_host, inference_http_port)
    model_version = ""
    logging.info("triton infer url: {}.".format(local_infer_url))
    if is_hg_model:
        inference_model_name = "{}_{}_inference".format(model_name, str(inference_engine))
    else:
        inference_model_name = model_name
    triton_client = http_client.InferenceServerClient(url=local_infer_url, verbose=False)
    wait_count = 0
    while True:
        if not triton_client.is_model_ready(
                model_name=inference_model_name, model_version=model_version
        ):
            logging.info(f"model {model_name} not yet ready")
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


def run_http_inference_with_curl_request(inference_url, inference_input_list, inference_output_list,
                                         inference_type=None):
    model_inference_result = {}
    model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
    print("inference_url: {}".format(inference_url))
    print("inference_input_list: {}".format(inference_input_list))
    if inference_output_list == []:
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

    model_file = "/home/fedml/fedml-client/fedml/models/theta_rec_auc_81_single_label/theta_rec_auc_81_single_label"
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
