
import logging
import os
import pickle
import platform
import shutil
import time
import requests
import torch
import torch.nn
import tritonclient.http as http_client

import collections.abc

for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from fedml.cli.model_deployment.device_client_constants import ClientConstants


def start_deployment(end_point_id, model_id, model_version,
                     model_storage_local_path, model_bin_file, inference_model_name, inference_engine,
                     inference_http_port, inference_grpc_port, inference_metric_port,
                     inference_use_gpu, inference_memory_size,
                     inference_convertor_image, inference_server_image,
                     infer_host, model_is_from_open, model_params,
                     model_from_open):
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

    logging.info("install nvidia docker...")

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
    if model_is_from_open:
        logging.info("convert the onnx model when the mode is from the MLOps platform...")

        logging.info("Input size {}, input types {}".format(model_params["input_size"],
                                                            model_params["input_types"]))

        running_model_name = ClientConstants.get_running_model_name(end_point_id, model_id,
                                                                    inference_model_name,
                                                                    model_version)
        if model_from_open is None:
            return running_model_name, "", model_version, {}, {}

        with open(model_bin_file, 'rb') as model_pkl_file:
            open_model_params = pickle.load(model_pkl_file)
            model_from_open.load_state_dict(open_model_params)
            model_from_open.eval()

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
            os.makedirs(onnx_model_path)
        onnx_model_path = os.path.join(onnx_model_path, "model.onnx")

        convert_model_to_onnx(model_from_open, onnx_model_path, dummy_input_list, input_size)
    else:
        logging.info("convert the onnx model when the mode is from the general PyTorch...")
        logging.info("Input size {}, input types {}".format(model_params["input_size"],
                                                            model_params["input_types"]))

        running_model_name = ClientConstants.get_running_model_name(end_point_id, model_id,
                                                                    inference_model_name,
                                                                    model_version)
        # configuration passed by user in the Cli
        model_location = os.path.join(model_storage_local_path, "fedml_model.bin")
        input_size = model_params["input_size"]
        input_types = model_params["input_types"]

        model = torch.jit.load(model_location)  # model def + params
        try:
            model.eval()
        except Exception as e:
            pass

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
            os.makedirs(onnx_model_path)
        onnx_model_path = os.path.join(onnx_model_path, "model.onnx")

        convert_model_to_onnx(model, onnx_model_path, dummy_input_list, input_size)

        # convert_model_container_name = "{}_{}_{}".format(ClientConstants.FEDML_CONVERT_MODEL_CONTAINER_NAME_PREFIX,
        #                                                  str(end_point_id),
        #                                                  str(model_id))
        # running_model_name = ClientConstants.get_running_model_name(end_point_id, model_id,
        #                                                             inference_model_name,
        #                                                             model_version)
        # model_storage_processed_path = ClientConstants.get_k8s_slave_host_dir(model_storage_local_path)
        # convert_model_cmd = "{}docker stop {}; {}docker rm {}; " \
        #                     "{}docker run --name {} --rm {} -v {}:/project " \
        #                     "{} bash -c \"cd /project && convert_model -m /project --name {} " \
        #                     "--backend {} --seq-len 16 128 128\"; exit". \
        #     format(sudo_prefix, convert_model_container_name, sudo_prefix, convert_model_container_name,
        #            sudo_prefix, convert_model_container_name, gpu_attach_cmd, model_storage_processed_path,
        #            inference_convertor_image, running_model_name,
        #            inference_engine)
        # logging.info("Convert the model to ONNX format: {}".format(convert_model_cmd))
        # logging.info("Now is converting the model to onnx, please wait...")
        # os.system(convert_model_cmd)
        # # convert_process = ClientConstants.exec_console_with_script(convert_model_cmd,
        # #                                                            should_capture_stdout=False,
        # #                                                            should_capture_stderr=False,
        # #                                                            no_sys_out_err=True)
        # log_deployment_result(end_point_id, model_id, convert_model_container_name,
        #                       ClientConstants.CMD_TYPE_CONVERT_MODEL, 0,
        #                       running_model_name, inference_engine, inference_http_port)

    # Move converted model to serving dir for inference
    logging.info("move converted model to serving dir for inference...")
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
    logging.info("prepare to run triton server...")
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


def get_model_info(model_name, inference_engine, inference_http_port, infer_host=None, is_hg_model=False):
    local_ip = ClientConstants.get_local_ip()
    if infer_host is not None and infer_host != "127.0.0.1":
        infer_url_host = infer_host
    else:
        infer_url_host = local_ip
    local_infer_url = "{}:{}".format(infer_url_host, inference_http_port)
    model_version = ""
    logging.info("triton infer url: {}.".format(local_infer_url))
    if is_hg_model:
        inference_model_name = "{}_{}_inference".format(model_name, inference_engine)
    else:
        inference_model_name = model_name
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


def run_http_inference_with_curl_request(inference_url, inference_input_list, inference_output_list):
    model_inference_result = {}
    model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
    model_inference_json = {
        "inputs": inference_input_list,
        "outputs": inference_output_list
    }

    response = requests.post(inference_url, headers=model_api_headers, json=model_inference_json)
    if response.status_code == 200:
        model_inference_result = response.json()

    return model_inference_result


def convert_model_to_onnx(
        torch_model, output_path: str, dummy_input_list, input_size: int, input_is_tensor=True
) -> None:
    from collections import OrderedDict
    import torch
    from torch.onnx import TrainingMode

    torch.onnx.export(torch_model,  # model being run
                      dummy_input_list if input_is_tensor else tuple(dummy_input_list),  # model input (or a tuple for multiple inputs)
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
        os.makedirs(onnx_model_dir)
    onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")

    convert_model_to_onnx(torch_model, onnx_model_path, dummy_input_list, input_size,
                          input_is_tensor=True)

    model_serving_dir = os.path.join(ClientConstants.get_model_cache_dir(),
                                     ClientConstants.FEDML_CONVERTED_MODEL_DIR_NAME)
    return model_serving_dir


if __name__ == "__main__":

    model_serving_dir = test_convert_pytorch_model_to_onnx("./sample-open-training-model-net",
                                                           "./sample-open-training-model",
                                                           "rec-model",
                                                           {"input_size": [[1,24], [1,2]],
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
