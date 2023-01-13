import argparse
import logging
import os
import platform
import shutil
import time

import numpy as np
import requests
import tritonclient.http as http_client

from fedml.cli.model_deployment.modelops_configs import ModelOpsConfigs
from fedml.cli.model_deployment.device_client_constants import ClientConstants


def start_deployment(end_point_id, model_id, model_version,
                     model_storage_local_path, inference_model_name, inference_engine,
                     inference_http_port, inference_grpc_port, inference_metric_port,
                     inference_use_gpu, inference_memory_size,
                     inference_convertor_image, inference_server_image,
                     infer_host):
    logging.info("Model deployment is starting...")

    gpu_attach_cmd = ""
    if inference_use_gpu is not None and inference_use_gpu != "":
        gpu_attach_cmd = "--gpus all"

    logging.info("Update docker environments...")

    sudo_prefix = "sudo "
    sys_name = platform.system()
    if sys_name == "Darwin":
        sudo_prefix = ""
        gpu_attach_cmd = ""

    # Check whether triton server is running.
    triton_server_is_running = False
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

    # Setup nvidia docker related packages.
    if not ClientConstants.is_running_on_k8s():
        if sys_name == "Linux":
            if not triton_server_is_running:
                os.system(sudo_prefix + "apt-get update")
                os.system(sudo_prefix + "apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin")
                os.system("distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
              && sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg;curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
              && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
                 sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                 sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
                os.system(sudo_prefix + "apt-get update")
                os.system(sudo_prefix + "apt-get install -y nvidia-docker2")
                os.system(sudo_prefix + "systemctl restart docker")

    # Convert models from pytorch to onnx format
    convert_model_container_name = "{}_{}_{}".format(ClientConstants.FEDML_CONVERT_MODEL_CONTAINER_NAME_PREFIX,
                                                     str(end_point_id),
                                                     str(model_id))
    running_model_name = ClientConstants.get_running_model_name(end_point_id, model_id,
                                                                inference_model_name,
                                                                model_version)
    model_storage_processed_path = ClientConstants.get_k8s_slave_host_dir(model_storage_local_path)
    convert_model_cmd = "{}docker stop {}; {}docker rm {}; " \
                        "{}docker run --name {} --rm {} -v {}:/project " \
                        "{} bash -c \"cd /project && convert_model -m /project --name {} " \
                        "--backend {} --seq-len 16 128 128\"; exit". \
        format(sudo_prefix, convert_model_container_name, sudo_prefix, convert_model_container_name,
               sudo_prefix, convert_model_container_name, gpu_attach_cmd, model_storage_processed_path,
               inference_convertor_image, running_model_name,
               inference_engine)
    logging.info("Convert the model to ONNX format: {}".format(convert_model_cmd))
    logging.info("Now is converting the model to onnx, please wait...")
    os.system(convert_model_cmd)
    # convert_process = ClientConstants.exec_console_with_script(convert_model_cmd,
    #                                                            should_capture_stdout=False,
    #                                                            should_capture_stderr=False,
    #                                                            no_sys_out_err=True)
    log_deployment_result(end_point_id, model_id, convert_model_container_name,
                          ClientConstants.CMD_TYPE_CONVERT_MODEL, 0,
                          running_model_name, inference_engine, inference_http_port)

    # Move converted model to serving dir for inference
    model_serving_dir = ClientConstants.get_model_serving_dir()
    if not os.path.exists(model_serving_dir):
        os.makedirs(model_serving_dir)
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

    # Run triton server
    if not triton_server_is_running and not ClientConstants.is_running_on_k8s():
        triton_server_cmd = "{}docker stop {}; {}docker rm {}; {}docker run --name {} {} -p{}:8000 " \
                            "-p{}:8001 -p{}:8002 " \
                            "--shm-size {} " \
                            "-v {}:/models {} " \
                            "bash -c \"pip install transformers && tritonserver " \
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

    inference_output_url, running_model_version, model_metadata, model_config = \
        get_model_info(running_model_name, inference_engine, inference_http_port, infer_host)
    logging.info("Deploy model successfully, inference url: {}, model metadata: {}, model config: {}".format(
        inference_output_url, model_metadata, model_config))

    return running_model_name, inference_output_url, model_version, model_metadata, model_config


def should_exit_logs(end_point_id, model_id, cmd_type, cmd_process_id, model_name, inference_engine, inference_port):
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
                get_model_info(model_name, inference_engine, inference_port)
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
                          inference_http_port):
    sudo_prefix = "sudo "
    sys_name = platform.system()
    if sys_name == "Darwin":
        sudo_prefix = ""

    last_out_logs = ""
    last_err_logs = ""
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

        if should_exit_logs(end_point_id, model_id, cmd_type, cmd_process_id,
                            inference_model_name, inference_engine, inference_http_port):
            break


def get_model_info(model_name, inference_engine, inference_http_port, infer_host=None):
    local_ip = ClientConstants.get_local_ip()
    if infer_host is not None and infer_host != "127.0.0.1":
        infer_url_host = infer_host
    else:
        infer_url_host = local_ip
    local_infer_url = "{}:{}".format(infer_url_host, inference_http_port)
    model_version = ""
    logging.info("triton infer url: {}.".format(local_infer_url))
    inference_model_name = "{}_{}_inference".format(model_name, inference_engine)
    triton_client = http_client.InferenceServerClient(url=local_infer_url, verbose=False)
    while True:
        if not triton_client.is_model_ready(
                model_name=inference_model_name, model_version=model_version
        ):
            logging.info(f"model {model_name} not yet ready")
            time.sleep(1)
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


def run_http_inference_with_lib_http_api(model_name, inference_http_port, batch_size,
                                         inference_input_data_list, host="localhost",
                                         inference_engine=ClientConstants.INFERENCE_ENGINE_TYPE_ONNX):
    local_infer_url = "{}:{}".format(host, inference_http_port)
    model_version = ClientConstants.INFERENCE_MODEL_VERSION
    inference_model_name = "{}_{}_inference".format(model_name, inference_engine)
    triton_client = http_client.InferenceServerClient(url=local_infer_url, verbose=False)
    while True:
        if not triton_client.is_model_ready(
                model_name=inference_model_name, model_version=model_version
        ):
            logging.info(f"model {model_name} not yet ready")
            time.sleep(1)
        else:
            break

    model_metadata = triton_client.get_model_metadata(model_name=inference_model_name, model_version=model_version)
    model_config = triton_client.get_model_config(model_name=inference_model_name, model_version=model_version)

    print("model metadata {}".format(model_metadata))
    inference_response_list = list()
    inference_input_list = model_metadata["inputs"]
    infer_item_count = 0
    inference_query_list = []

    input_data_np = np.asarray(inference_input_data_list * batch_size, dtype=object)

    for infer_input_item in inference_input_list:
        query_item = http_client.InferInput(name=infer_input_item["name"],
                                            shape=(batch_size,), datatype=infer_input_item["datatype"])
        query_item.set_data_from_numpy(input_data_np)
        inference_query_list.append(query_item)
        infer_item_count += 1

    inference_output_list = model_metadata["outputs"]
    infer_item_count = 0
    inference_result_list = []
    for infer_output_item in inference_output_list:
        result_item = http_client.InferRequestedOutput(name=infer_output_item["name"], binary_data=False)
        inference_result_list.append(result_item)
        infer_item_count += 1

    response = triton_client.infer(
        model_name=inference_model_name, model_version=model_version, inputs=inference_query_list,
        outputs=inference_result_list
    )

    for infer_output_item in inference_output_list:
        response_item = response.get_output(infer_output_item["name"])
        inference_response_list.append(response_item)
        print("response item {}".format(response_item))

    inference_response_dict = {"outputs": inference_response_list}
    print("return {}".format(inference_response_dict))
    return inference_response_dict


def run_http_inference_with_raw_http_request(self, inference_input_json, inference_input_data_list):
    inference_output_sample = {}

    inference_input_list = inference_input_json["inputs"]
    infer_item_count = 0
    inference_query_list = []
    for infer_input_item in inference_input_list:
        infer_input_item["parameters"] = {"binary_data_size": len(inference_input_data_list[infer_item_count])}
        inference_query_list.append(infer_input_item)
        infer_item_count += 1

    inference_output_list = inference_input_json["outputs"]
    infer_item_count = 0
    inference_result_list = []
    for infer_output_item in inference_output_list:
        infer_output_item["parameters"] = {"binary_data": False}
        inference_result_list.append(infer_output_item)
        infer_item_count += 1

    inference_input_data = bytearray()
    inference_input_data.append(inference_query_list)
    inference_input_data.append(inference_input_data_list)

    inference_headers = {'Content-Type': 'application/binary', 'Connection': 'close'}
    _, cert_path = ModelOpsConfigs.get_instance(self.args).get_request_params()
    if cert_path is not None:
        try:
            requests.session().verify = cert_path
            response = requests.post(
                self.log_server_url, data=inference_input_data, verify=True, headers=inference_headers
            )
        except requests.exceptions.SSLError as err:
            ModelOpsConfigs.install_root_ca_file()
            response = requests.post(
                self.log_server_url, data=inference_input_data, verify=True, headers=inference_headers
            )
    else:
        response = requests.post(self.log_server_url, data=inference_input_data, headers=inference_headers)
    if response.status_code != 200:
        pass
    else:
        resp_data = response.json()
        inference_output_sample = resp_data

    return inference_output_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cf", "-c", help="config file")
    parser.add_argument("--role", "-r", type=str, default="client", help="role")
    parser.add_argument("--model_storage_local_path", "-url", type=str, default="/home/ubuntu",
                        help="model storage local path")
    parser.add_argument("--inference_model_name", "-n", type=str, default="fedml-model",
                        help="inference model name")
    parser.add_argument("--inference_engine", "-engine", type=str, default="ONNX", help="inference engine")
    parser.add_argument("--inference_http_port", "-http", type=int, default=8000, help="inference http port")
    parser.add_argument("--inference_grpc_port", "-gprc", type=int, default=8001, help="inference grpc port")
    parser.add_argument("--inference_metric_port", "-metric", type=int, default=8002, help="inference metric port")
    parser.add_argument("--inference_use_gpu", "-gpu", type=str, default="gpu", help="inference use gpu")
    parser.add_argument("--inference_memory_size", "-mem", type=str, default="256m", help="inference memory size")
    parser.add_argument("--inference_convertor_image", "-convertor", type=str,
                        default=ClientConstants.INFERENCE_CONVERTOR_IMAGE, help="inference convertor image")
    parser.add_argument("--inference_server_image", "-server", type=str,
                        default=ClientConstants.INFERENCE_SERVER_IMAGE, help="inference server image")
    args = parser.parse_args()
    args.user = args.user

    pip_source_dir = os.path.dirname(__file__)
    __running_model_name, __inference_output_url, __model_version, __model_metadata, __model_config = \
        start_deployment(
            args.model_storage_local_path,
            args.inference_model_name,
            args.inference_engine,
            args.inference_http_port,
            args.inference_grpc_port,
            args.inference_metric_port,
            args.inference_use_gpu,
            args.inference_memory_size,
            args.inference_convertor_image,
            args.inference_server_image)
    print("Model deployment results, running model name: {}, url: {}, model metadata: {}, model config: {}".format(
        __running_model_name, __inference_output_url, __model_metadata, __model_config))
