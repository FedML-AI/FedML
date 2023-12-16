import argparse
import json
import logging
import os
import time
import uuid
from os.path import expanduser

import fedml
from fedml.api.modules import model
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.job_monitor import JobMonitor
from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
from fedml.computing.scheduler.model_scheduler.device_http_inference_protocol import FedMLHttpInference
from fedml.computing.scheduler.model_scheduler.device_http_proxy_inference_protocol import FedMLHttpProxyInference
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from fedml.computing.scheduler.model_scheduler import device_client_constants


def test_model_create_push(config_version="release"):
    cur_dir = os.path.dirname(__file__)
    model_config = os.path.join(cur_dir, "llm_deploy", "serving.yaml")
    model_name = f"test_model_{str(uuid.uuid4())}"
    fedml.set_env_version(config_version)
    model.create(model_name, model_config=model_config)
    model.push(
        model_name, api_key="10e87dd6d6574311a80200455e4d9b30",
        tag_list=[{"tagId": 147, "parentId": 3, "tagName": "LLM"}])


def test_cleanup_model_monitor_process():
    sys_utils.cleanup_model_monitor_processes(
        1627, "ep-1124-304-13ad33", "", "", "")


def test_log_endpoint_status():
    fedml.set_env_version("dev")

    endpoint_id = 1682
    fedml.set_env_version("dev")
    fedml.mlops.log_endpoint_status(
        endpoint_id, device_client_constants.ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)


def test_show_hide_mlops_console_logs(args):
    args.log_file_dir = ClientConstants.get_log_file_dir()
    args.run_id = 0
    args.role = "client"
    client_ids = list()
    client_ids.append(111)
    args.client_id_list = json.dumps(client_ids)
    setattr(args, "using_mlops", True)
    MLOpsRuntimeLog.get_instance(args).init_logs(show_stdout_log=False)
    print("log 1")
    MLOpsRuntimeLog.get_instance(args).enable_show_log_to_stdout()
    print("log 2")
    MLOpsRuntimeLog.get_instance(args).enable_show_log_to_stdout(enable=False)
    print("log 3")


def test_log_lines_to_mlops():
    fedml.mlops.log_run_log_lines(
        1685, 0, ["failed to upload logs4"], log_source="MODEL_END_POINT")


def test_unify_inference():
    # Test inference
    from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
    mqtt_config = dict()
    mqtt_config["BROKER_HOST"] = "mqtt-test.fedml.ai"
    mqtt_config["BROKER_PORT"] = 1883
    mqtt_config["MQTT_USER"] = "admin"
    mqtt_config["MQTT_PWD"] = "2Dsrd48UqLQZFsfbW9uZ"
    mqtt_config["MQTT_KEEPALIVE"] = 180
    JobRunnerUtils.get_instance().mqtt_config = mqtt_config
    model_url = "http://127.0.0.1:32768/predict"
    input_list = {"messages": [{"role": "user", "content": "What is a good cure for hiccups?"}]}
    JobRunnerUtils.get_instance().inference(
        244, 350, model_url, input_list, [], )


def test_http_inference():
    # Test http and http proxy inference
    endpoint_id = 383
    inference_url = "http://127.0.0.1:10000/predict"
    input = {"messages": [{"role": "user", "content": "What is a good cure for hiccups?"}]}
    output = []
    print(FedMLHttpProxyInference.run_http_proxy_inference_with_request(
        endpoint_id, inference_url, input, output))

    print(FedMLHttpInference.run_http_inference_with_curl_request(inference_url, input, output))


def test_http_inference_with_stream_mode():
    # Test http and http proxy inference
    endpoint_id = 383
    inference_url = "http://127.0.0.1:10000/predict"
    input = {
        "stream": True, "messages": [{"role": "system", "content": "You are a helpful assistant."},
                                     {"role": "user", "content": "Who won the world series in 2020?"},
                                     {"role": "assistant",
                                      "content": "The Los Angeles Dodgers won the World Series in 2020."},
                                     {"role": "user", "content": "Where was it played?"}],
        "model": "mythomax-l2-13b-openai-endpoint01"}
    output = []
    print(FedMLHttpProxyInference.run_http_proxy_inference_with_request(
        endpoint_id, inference_url, input, output))

    print(FedMLHttpInference.run_http_inference_with_curl_request(inference_url, input, output))


def test_create_container():
    import docker
    client = docker.from_env()
    inference_image_name = "fedml/fedml-default-inference-backend"
    default_server_container_name = "fedml-test-docker"
    port_inside_container = 2345
    usr_indicated_worker_port = None
    new_container = client.api.create_container(
        image=inference_image_name,
        name=default_server_container_name,
        # volumes=volumns,
        ports=[port_inside_container],  # port open inside the container
        entrypoint=["sleep", '1000'],
        # environment=environment,
        host_config=client.api.create_host_config(
            # binds=binds,
            port_bindings={
                port_inside_container: usr_indicated_worker_port  # Could be either None or a port number
            },
            # device_requests=device_requests,
            # mem_limit = "8g",   # Could also be configured in the docker desktop setting
        ),
        detach=True,
        # command=entry_cmd if enable_custom_image else None
    )
    client.api.start(container=new_container.get("Id"))

    # Get the port allocation
    cnt = 0
    while True:
        cnt += 1
        try:
            if usr_indicated_worker_port is not None:
                inference_http_port = usr_indicated_worker_port
                break
            else:
                # Find the random port
                port_info = client.api.port(new_container.get("Id"), port_inside_container)
                inference_http_port = port_info[0]["HostPort"]
                logging.info("inference_http_port: {}".format(inference_http_port))
                break
        except:
            if cnt >= 5:
                raise Exception("Failed to get the port allocation")
            time.sleep(3)


def test_get_endpoint_logs():
    JobMonitor.get_instance().monitor_endpoint_logs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", "-v", type=str, default="dev")
    parser.add_argument("--log_file_dir", "-l", type=str, default="~")
    args = parser.parse_args()

    print("Hi everyone, I am testing the model cli.\n")

    logging.getLogger().setLevel(logging.INFO)
    fedml.set_env_version("dev")

    test_get_endpoint_logs()

    time.sleep(1000000)
