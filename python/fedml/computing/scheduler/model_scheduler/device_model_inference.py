import logging
import time
import traceback
from urllib.parse import urlparse
import os

from fastapi import FastAPI, Request
from fedml.computing.scheduler.model_scheduler.device_model_deployment import run_http_inference_with_curl_request
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.computing.scheduler.model_scheduler.device_model_monitor import FedMLModelMetrics
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.computing.scheduler.model_scheduler.device_mqtt_inference_protocol import FedMLMqttInfernce
from fedml.computing.scheduler.model_scheduler.device_http_proxy_inference_protocol import FedMLHttpProxyInfernce
from fedml.computing.scheduler.comm_utils import sys_utils

from pydantic import BaseSettings


class Settings(BaseSettings):
    redis_addr: str
    redis_port: str
    redis_password: str
    end_point_name: str
    model_name: str
    model_version: str
    model_infer_url: str
    version: str
    use_mqtt_inference: bool
    use_worker_gateway: bool
    ext_info: str


settings = Settings()

# class settings:
#     redis_addr = "127.0.0.1"
#     redis_port = 6379
#     redis_password = "fedml_default"
#     end_point_name = ""
#     model_name = ""
#     model_version = ""
#     model_infer_url = "127.0.0.1"
#     version = "dev"
#     use_mqtt_inference = False
#     use_worker_gateway = False
#     ext_info = "2b34303961245c4f175f2236282d7a272c040b0904747579087f6a760112030109010c215d54505707140005190a051c347f365c4a430c020a7d39120e26032a78730f797f7c031f0901657e75"


api = FastAPI()


@api.get('/')
def root():
    return {'message': 'FedML Federated Inference Service!'}


@api.post('/api/v1/predict')
async def predict(request: Request):
    # Get json data
    input_json = await request.json()
    end_point_id = input_json.get("end_point_id", None)

    # Get header
    header = request.headers

    return _predict(end_point_id, input_json, header)


@api.post('/inference/{end_point_id}')
async def predict_with_end_point_id(end_point_id, request: Request):
    # Get json data
    input_json = await request.json()

    # Get header
    header = request.headers

    return _predict(end_point_id, input_json, header)


def _predict(end_point_id, input_json, header=None):
    in_end_point_id = end_point_id
    in_end_point_name = input_json.get("end_point_name", None)
    in_model_name = input_json.get("model_name", None)
    in_model_version = input_json.get("model_version", None)
    in_end_point_token = input_json.get("token", None)
    in_return_type = "default"
    if header is not None:
        in_return_type = header.get("Accept", "default")

    if in_model_version is None:
        in_model_version = "latest"

    print("Inference json: {}".format(input_json))

    start_time = time.time_ns()

    # Authenticate request token
    inference_response = {}
    if auth_request_token(in_end_point_id, in_end_point_name, in_model_name, in_end_point_token):
        # Found idle inference device
        idle_device, end_point_id, model_id, model_name, model_version, inference_host, inference_output_url = \
            found_idle_inference_device(in_end_point_id, in_end_point_name, in_model_name, in_model_version)

        # Start timing for model metrics
        model_metrics = FedMLModelMetrics(end_point_id, in_end_point_name,
                                          model_id, in_model_name, model_version,
                                          settings.model_infer_url,
                                          settings.redis_addr, settings.redis_port, settings.redis_password,
                                          version=settings.version)
        model_metrics.set_start_time(start_time)

        # Send inference request to idle device
        print("inference url {}.".format(inference_output_url))
        if inference_output_url != "":
            input_list = input_json["inputs"]
            stream_flag = input_json.get("stream", False)
            input_list["stream"] = input_list.get("stream", stream_flag)
            output_list = input_json.get("outputs", [])
            inference_response = send_inference_request(
                idle_device, end_point_id, inference_output_url, input_list, output_list, inference_type=in_return_type)

        # Calculate model metrics
        try:
            model_metrics.calc_metrics(end_point_id, in_end_point_name,
                                       model_id, model_name, model_version,
                                       inference_output_url, idle_device)
        except Exception as e:
            print("Calculate Inference Metrics Exception: {}".format(traceback.format_exc()))
            pass

        logging_inference_request(input_json, inference_response)

        return inference_response

    else:
        inference_response = {"error": True, "message": "token is not valid."}
        logging_inference_request(input_json, inference_response)
        return inference_response

    return inference_response


def found_idle_inference_device(end_point_id, end_point_name, in_model_name, in_model_version):
    idle_device = ""
    model_name = ""
    model_id = ""
    inference_host = ""
    inference_output_url = ""
    # Found idle device (TODO: optimize the algorithm to search best device for inference)
    FedMLModelCache.get_instance().set_redis_params(settings.redis_addr, settings.redis_port, settings.redis_password)
    payload, idle_device = FedMLModelCache.get_instance(settings.redis_addr, settings.redis_port). \
        get_idle_device(end_point_id, end_point_name, in_model_name, in_model_version)
    if payload is not None:
        print("found idle deployment result {}".format(payload))
        deployment_result = payload
        model_name = deployment_result["model_name"]
        model_version = deployment_result["model_version"]
        model_id = deployment_result["model_id"]
        end_point_id = deployment_result["end_point_id"]
        inference_output_url = deployment_result["model_url"]
        url_parsed = urlparse(inference_output_url)
        inference_host = url_parsed.hostname

    return idle_device, end_point_id, model_id, model_name, model_version, inference_host, inference_output_url


def send_inference_request(idle_device, endpoint_id, inference_url, input_list, output_list, inference_type="default"):
    try:
        response_ok, inference_response = run_http_inference_with_curl_request(
            inference_url, input_list, output_list, inference_type=inference_type)
        if response_ok:
            print("Use http inference.")
            return inference_response
        print("Use http inference failed.")

        http_proxy_inference = FedMLHttpProxyInfernce()
        response_ok, inference_response = http_proxy_inference.run_http_proxy_inference_with_request(
            endpoint_id, inference_url, input_list, output_list, inference_type=inference_type)
        if response_ok:
            print("Use http proxy inference.")
            return inference_response

        connect_str = "@FEDML@"
        random_out = sys_utils.random2(settings.ext_info, "FEDML@9999GREAT")
        config_list = random_out.split(connect_str)
        agent_config = dict()
        agent_config["mqtt_config"] = dict()
        agent_config["mqtt_config"]["BROKER_HOST"] = config_list[0]
        agent_config["mqtt_config"]["BROKER_PORT"] = int(config_list[1])
        agent_config["mqtt_config"]["MQTT_USER"] = config_list[2]
        agent_config["mqtt_config"]["MQTT_PWD"]  = config_list[3]
        agent_config["mqtt_config"]["MQTT_KEEPALIVE"] = int(config_list[4])
        mqtt_inference = FedMLMqttInfernce(agent_config=agent_config, run_id=endpoint_id)
        inference_response = mqtt_inference.run_mqtt_inference_with_request(
            idle_device, endpoint_id, inference_url, input_list, output_list, inference_type=inference_type)

        print("Use mqtt inference.")
        return inference_response
    except Exception as e:
        logging.info("Inference Exception: {}".format(traceback.format_exc()))
        pass

    return {}


def auth_request_token(end_point_id, end_point_name, model_name, token):
    if token is None:
        return False

    FedMLModelCache.get_instance().set_redis_params(settings.redis_addr, settings.redis_port, settings.redis_password)
    cached_token = FedMLModelCache.get_instance(settings.redis_addr, settings.redis_port). \
        get_end_point_token(end_point_id, end_point_name, model_name)
    if cached_token is not None and cached_token == token:
        return True

    return False


def logging_inference_request(request, response):
    try:
        log_dir = ServerConstants.get_log_file_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        inference_log_file = os.path.join(log_dir, "inference.log")
        with open(inference_log_file, "a") as f:
            f.writelines([f"request: {request}, response: {response}\n"])
    except Exception as ex:
        print("failed to log inference request and response to file.")


if __name__ == "__main__":
    import uvicorn
    port = 2204
    uvicorn.run(api, host="0.0.0.0", port=port)
