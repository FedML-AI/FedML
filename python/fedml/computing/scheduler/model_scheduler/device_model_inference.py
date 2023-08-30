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


settings = Settings()
api = FastAPI()


@api.get('/')
def root():
    return {'message': 'FedML Federated Inference Service!'}


@api.post('/api/v1/predict')
async def predict(request: Request):
    # Get json data
    input_json = await request.json()
    in_end_point_name = input_json.get("end_point_name", None)
    in_model_name = input_json.get("model_name", None)
    in_model_version = input_json.get("model_version", None)
    in_end_point_token = input_json.get("token", None)
    if in_model_version is None:
        in_model_version = "latest"

    print("Inference json: {}".format(input_json))

    start_time = time.time_ns()

    # Authenticate request token
    inference_response = {}
    if auth_request_token(in_end_point_name, in_model_name, in_end_point_token):
        # Found idle inference device
        idle_device, end_point_id, model_id, model_name, model_version, inference_host, inference_output_url = \
            found_idle_inference_device(in_end_point_name, in_model_name, in_model_version)

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
            output_list = input_json["outputs"]
            inference_response = send_inference_request(inference_output_url, input_list, output_list)

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


def found_idle_inference_device(end_point_name, in_model_name, in_model_version):
    idle_device = ""
    model_name = ""
    model_id = ""
    end_point_id = ""
    inference_host = ""
    inference_output_url = ""
    # Found idle device (TODO: optimize the algorithm to search best device for inference)
    FedMLModelCache.get_instance().set_redis_params(settings.redis_addr, settings.redis_port, settings.redis_password)
    payload, idle_device = FedMLModelCache.get_instance(settings.redis_addr, settings.redis_port). \
        get_idle_device(end_point_name, in_model_name, in_model_version)
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


def send_inference_request(inference_url, input_list, output_list):
    try:
        inference_response = run_http_inference_with_curl_request(inference_url, input_list, output_list)
        return inference_response
    except Exception as e:
        logging.info("Inference Exception: {}".format(traceback.format_exc()))
        pass

    return {}


def auth_request_token(end_point_name, model_name, token):
    if token is None:
        return False

    FedMLModelCache.get_instance().set_redis_params(settings.redis_addr, settings.redis_port, settings.redis_password)
    cached_token = FedMLModelCache.get_instance(settings.redis_addr, settings.redis_port). \
        get_end_point_token(end_point_name, model_name)
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

