import logging
import time
import traceback
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fedml.cli.model_deployment.device_model_deployment import run_http_inference_with_lib_http_api, \
    run_http_inference_with_raw_http_request, run_http_inference_with_lib_http_api_with_image_data
from fedml.cli.model_deployment.device_client_constants import ClientConstants
from fedml.cli.model_deployment.device_server_constants import ServerConstants
from fedml.cli.model_deployment.device_model_monitor import FedMLModelMetrics
from fedml.cli.model_deployment.device_model_cache import FedMLModelCache

import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    redis_addr: str
    redis_port: str
    redis_password: str
    end_point_id: str
    model_id: str
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
    in_end_point_id = input_json.get("end_point_id", None)
    in_model_id = input_json.get("model_id", None)
    in_model_name = input_json.get("model_name", None)
    in_model_version = input_json.get("model_version", None)
    in_end_point_token = input_json.get("token", None)
    if in_end_point_id is None or in_end_point_id == "":
        in_end_point_id = settings.end_point_id
        in_model_id = settings.model_id
        in_model_name = settings.model_name
        in_model_version = settings.model_version

    print("Inference json: {}".format(input_json))
    print(f"Current end point id {in_end_point_id}.")

    # Start timing for model metrics
    model_metrics = FedMLModelMetrics(in_end_point_id, in_model_id,
                                      in_model_name, settings.model_infer_url,
                                      settings.redis_addr, settings.redis_port, settings.redis_password,
                                      version=settings.version)
    model_metrics.set_start_time()

    # Authenticate request token
    inference_response = {}
    has_requested_inference = False
    if auth_request_token(in_end_point_id, in_end_point_token):
        # Found idle inference device
        idle_device, model_id, model_name, inference_host, inference_output_url = \
            found_idle_inference_device(in_end_point_id, in_model_id)

        # Send inference request to idle device
        if inference_output_url != "":
            input_data = input_json.get("data", "SampleData")
            input_data_list = list()
            input_data_list.append(str(input_data))
            inference_response = send_inference_request(idle_device, model_name, inference_host,
                                                        inference_output_url, input_json, input_data_list)
            has_requested_inference = True
    else:
        inference_response = {"error": True, "message": "token is not valid."}
        return inference_response

    if not has_requested_inference:
        return inference_response

    # Calculate model metrics
    model_metrics.calc_metrics(model_id, model_name, in_end_point_id, inference_output_url)

    return inference_response


def found_idle_inference_device(end_point_id, in_model_id):
    idle_device = ""
    model_name = ""
    inference_host = ""
    model_id = in_model_id
    inference_output_url = ""
    inference_port = ServerConstants.INFERENCE_HTTP_PORT
    # Found idle device (TODO: optimize the algorithm to search best device for inference)
    FedMLModelCache.get_instance().set_redis_params(settings.redis_addr, settings.redis_port, settings.redis_password)
    payload = FedMLModelCache.get_instance(settings.redis_addr, settings.redis_port).get_idle_device(end_point_id,
                                                                                                     in_model_id)
    if payload is not None:
        print("found idle deployment result {}".format(payload))
        deployment_result = payload
        model_name = deployment_result["model_name"]
        model_id = deployment_result["model_id"]
        inference_output_url = deployment_result["model_url"]
        inference_port = deployment_result["port"]
        url_parsed = urlparse(inference_output_url)
        inference_host = url_parsed.hostname

    return idle_device, model_id, model_name, inference_host, inference_output_url


def send_inference_request(device, model_name, inference_host, inference_url, json_req, input_data_list=None):
    try:
        if input_data_list[0].startswith("http://") or input_data_list[0].startswith("https://") or \
                input_data_list[0].startswith("file://"):
            inference_response = run_http_inference_with_lib_http_api_with_image_data(model_name,
                                                                                      ClientConstants.INFERENCE_HTTP_PORT,
                                                                                      1,
                                                                                      input_data_list,
                                                                                      inference_host)
        else:
            inference_response = run_http_inference_with_lib_http_api(model_name,
                                                                      ClientConstants.INFERENCE_HTTP_PORT,
                                                                      1,
                                                                      input_data_list,
                                                                      inference_host)
        return inference_response
    except Exception as e:
        logging.info("Inference Exception: {}".format(traceback.format_exc()))
        pass

    return {}


def run_inference(json_req, bin_data=None, host="localhost"):
    model_name = json_req["model_name"]
    infer_data = json_req["infer_data"]
    predication_result = run_http_inference_with_lib_http_api(model_name,
                                                              ClientConstants.INFERENCE_HTTP_PORT, 1, infer_data,
                                                              host)
    return predication_result


def auth_request_token(end_point_id, token):
    if token is None:
        return False

    FedMLModelCache.get_instance().set_redis_params(settings.redis_addr, settings.redis_port, settings.redis_password)
    cached_token = FedMLModelCache.get_instance(settings.redis_addr, settings.redis_port). \
        get_end_point_token(end_point_id)
    if cached_token is not None and cached_token == token:
        return True

    return False
