import time
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fedml.cli.model_deployment.device_model_deployment import run_http_inference_with_lib_http_api, \
    run_http_inference_with_raw_http_request
from fedml.cli.model_deployment.device_client_constants import ClientConstants
from fedml.cli.model_deployment.device_server_constants import ServerConstants
from fedml.cli.model_deployment.device_model_monitor import FedMLModelMetrics
from fedml.cli.model_deployment.device_model_cache import FedMLModelCache

import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    end_point_id: str
    model_id: str
    model_name: str
    model_version: str
    model_infer_url: str


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
    if in_end_point_id is None or in_end_point_id == "":
        in_end_point_id = settings.end_point_id
        in_model_id = settings.model_id
        in_model_name = settings.model_name
        in_model_version = settings.model_version

    print("Inference json: {}".format(input_json))
    print(f"Current end point id {in_end_point_id}.")

    model_metrics = FedMLModelMetrics(in_end_point_id, in_model_id,
                                      in_model_name, settings.model_infer_url)
    model_metrics.set_start_time()

    # Found idle inference device
    idle_device, model_id, model_name, inference_host, inference_output_url = \
        found_idle_inference_device(in_end_point_id, in_model_id)

    # Send inference request to idle device
    inference_response = {}
    if inference_output_url != "":
        input_data = input_json.get("data", "SampleData")
        input_data_list = list()
        input_data_list.append(str(input_data))
        inference_response = send_inference_request(idle_device, model_name, inference_host,
                                                    inference_output_url, input_json, input_data_list)

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
    payload = FedMLModelCache.get_instance().get_idle_device(end_point_id, in_model_id)
    if payload != {}:
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
    inference_response = run_http_inference_with_lib_http_api(model_name,
                                                              ClientConstants.INFERENCE_HTTP_PORT,
                                                              1,
                                                              input_data_list,
                                                              inference_host)
    return inference_response


def run_inference(json_req, bin_data=None, host="localhost"):
    model_name = json_req["model_name"]
    infer_data = json_req["infer_data"]
    predication_result = run_http_inference_with_lib_http_api(model_name,
                                                              ClientConstants.INFERENCE_HTTP_PORT, 1, infer_data,
                                                              host)
    return predication_result
