import argparse
import logging
import os
import platform
import time

import requests
import tritonclient.http as http_client

from fedml.cli.model_deployment.modelops_configs import ModelOpsConfigs
from fedml.cli.model_deployment.device_client_constants import ClientConstants


def start_deployment(model_storage_local_path, inference_model_dir_name, inference_engine,
                     inference_http_port, inference_grpc_port, inference_metric_port,
                     inference_use_gpu, inference_memory_size,
                     inference_convertor_image, inference_server_image):
    inference_output_url = ""
    model_metadata = {}
    model_config = {}
    model_version = ClientConstants.INFERENCE_MODEL_VERSION

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
    if sys_name == "Linux":
        os.system(sudo_prefix + "apt-get update")
        os.system(sudo_prefix + "apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin")
        os.system(sudo_prefix + "apt-get install -y nvidia-docker2")
        os.system(sudo_prefix + "systemctl restart docker")

    convert_model_cmd = "{}docker run -it --rm {} -v {}:/project {} " \
                        "bash -c \"cd /project && convert_model -m /project " \
                        "--backend {} --seq-len 16 128 128\"".format(sudo_prefix,
                                                                     gpu_attach_cmd,
                                                                     model_storage_local_path,
                                                                     inference_convertor_image,
                                                                     inference_engine)
    logging.info("Convert the model to ONNX format: {}".format(convert_model_cmd))
    os.system(convert_model_cmd)
    # convert_process = ClientConstants.exec_console_with_script(convert_model_cmd, should_capture_stdout=True,
    #                                                            should_capture_stderr=True)
    # ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(convert_process)
    # if out is not None:
    #     out_str = out.decode(encoding="utf-8")
    # if err is not None:
    #     err_str = err.decode(encoding="utf-8")

    triton_server_cmd = "{}docker run -it {} -p{}:8000 " \
                        "-p{}:8001 -p{}:8002 " \
                        "--shm-size {} " \
                        "-v {}/triton_models:/models {} " \
                        "bash -c \"pip install transformers && tritonserver " \
                        "--model-repository=/models\"".format(sudo_prefix,
                                                              gpu_attach_cmd,
                                                              inference_http_port,
                                                              inference_grpc_port,
                                                              inference_metric_port,
                                                              inference_memory_size,
                                                              model_storage_local_path,
                                                              inference_server_image)
    logging.info("Run triton inference server: {}".format(triton_server_cmd))
    os.system(triton_server_cmd)
    inference_output_url, model_version, model_metadata, model_config = \
        get_model_info(inference_model_dir_name, inference_http_port)
    logging.info("Deploy model successfully, inference url: {}, model metadata: {}, model config: {}".format(
        inference_output_url, model_metadata, model_config))
    # triton_process = ClientConstants.exec_console_with_script(triton_server_cmd,
    #                                                           should_capture_stdout=True,
    #                                                           should_capture_stderr=True)
    # ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(triton_process)
    # if out is not None:
    #     out_str = out.decode(encoding="utf-8")
    #     if str(out_str).find(ClientConstants.INFERENCE_SERVER_STARTED_TAG) != -1:
    #         inference_output_url, model_version, model_metadata, model_config = \
    #             get_model_info(inference_model_dir_name, inference_http_port)
    #         logging.info("Deploy model successfully, inference url: {}, model metadata: {}, model config: {}".format(
    #             inference_output_url, model_metadata, model_config))
    # elif err is not None:
    #     err_str = err.decode(encoding="utf-8")
    #     logging.error("Deploy model failed: {}".format(err_str))

    return inference_output_url, model_version, model_metadata, model_config


def get_model_info(model_name, inference_http_port):
    ip = ClientConstants.get_local_ip()
    local_infer_url = "{}:{}".format(ip, inference_http_port)
    model_version = ""
    triton_client = http_client.InferenceServerClient(url=local_infer_url, verbose=False)
    while True:
        if not triton_client.is_model_ready(
                model_name=model_name, model_version=model_version
        ):
            logging.info(f"model {model_name} not yet ready")
            time.sleep(1)
        else:
            break

    model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)
    version_list = model_metadata.get("versions", None)
    if version_list is not None and len(version_list) > 0:
        model_version = version_list[0]
    else:
        model_version = ClientConstants.INFERENCE_MODEL_VERSION

    inference_output_url = "{}/{}/models/{}/versions/{}/infer".format(local_infer_url,
                                                                      ClientConstants.INFERENCE_INFERENCE_SERVER_VERSION,
                                                                      model_name,
                                                                      model_version)

    return inference_output_url, model_version, model_metadata, model_config


def run_http_inference_with_lib_http_api(model_name, inference_http_port, batch_size,
                                         inference_input_data_list, host="localhost"):
    local_infer_url = "{}:{}".format(host, inference_http_port)
    model_version = ClientConstants.INFERENCE_MODEL_VERSION
    triton_client = http_client.InferenceServerClient(url=local_infer_url, verbose=False)
    while True:
        if not triton_client.is_model_ready(
                model_name=model_name, model_version=model_version
        ):
            logging.info(f"model {model_name} not yet ready")
            time.sleep(1)
        else:
            break

    model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

    inference_output_sample = {}
    inference_input_list = model_metadata["inputs"]
    infer_item_count = 0
    inference_query_list = []
    for infer_input_item in inference_input_list:
        query_item = http_client.InferInput(name=infer_input_item["name"],
                                                  shape=(batch_size,), datatype=infer_input_item["data_type"])
        query_item.set_data_from_numpy(inference_input_data_list[infer_item_count])
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
        model_name=model_name, model_version=model_version, inputs=inference_query_list, outputs=inference_result_list
    )

    for infer_output_item in inference_output_list:
        inference_output_sample[infer_output_item["name"]] = response.as_numpy(infer_output_item["name"])

    return inference_output_sample


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
    parser.add_argument("--inference_model_dir_name", "-dir", type=str, default="fedml-model",
                        help="inference model dir name")
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
    inference_output_url, model_version, model_metadata, model_config = start_deployment(args.model_storage_local_path,
                                                                          args.inference_model_dir_name,
                                                                          args.inference_engine,
                                                                          args.inference_http_port,
                                                                          args.inference_grpc_port,
                                                                          args.inference_metric_port,
                                                                          args.inference_use_gpu,
                                                                          args.inference_memory_size,
                                                                          args.inference_convertor_image,
                                                                          args.inference_server_image)
    print("Model deployment results, url: {}, model metadata: {}, model config: {}".format(
        inference_output_url, model_metadata, model_config))

