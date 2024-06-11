import argparse
import json
import logging
import time
import traceback
import os

from typing import Any, Mapping, MutableMapping, Union
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse

import fedml
from fedml.api.modules.constants import ModuleConstants
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.computing.scheduler.model_scheduler.device_http_inference_protocol import FedMLHttpInference
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.computing.scheduler.model_scheduler.device_model_monitor import FedMLModelMetrics
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.computing.scheduler.model_scheduler.device_mqtt_inference_protocol import FedMLMqttInference
from fedml.computing.scheduler.model_scheduler.device_http_proxy_inference_protocol import FedMLHttpProxyInference
from fedml.core.mlops.mlops_configs import MLOpsConfigs
from fedml.core.mlops import MLOpsRuntimeLog, MLOpsRuntimeLogDaemon


class Settings:
    server_name = "DEVICE_INFERENCE_GATEWAY"
    fedml.load_env()
    redis_addr = os.getenv(ModuleConstants.ENV_FEDML_INFER_REDIS_ADDR, SchedulerConstants.REDIS_ADDR)
    redis_port = os.getenv(ModuleConstants.ENV_FEDML_INFER_REDIS_PORT, SchedulerConstants.REDIS_PORT)
    redis_password = os.getenv(ModuleConstants.ENV_FEDML_INFER_REDIS_PASSWORD, SchedulerConstants.REDIS_PASSWORD)
    model_infer_host = os.getenv(ModuleConstants.ENV_FEDML_INFER_HOST, SchedulerConstants.REDIS_INFER_HOST)
    version = fedml.get_env_version()
    mqtt_config = MLOpsConfigs.fetch_mqtt_config()


api = FastAPI()

FEDML_MODEL_CACHE = FedMLModelCache.get_instance()
FEDML_MODEL_CACHE.set_redis_params(redis_addr=Settings.redis_addr,
                                   redis_port=Settings.redis_port,
                                   redis_password=Settings.redis_password)


@api.middleware("http")
async def auth_middleware(request: Request, call_next):
    if "/inference" in request.url.path or "/api/v1/predict" in request.url.path:
        try:
            # Attempt to parse the JSON body.
            request_json = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                {"error": True, "message": "Invalid JSON."},
                status_code=status.HTTP_400_BAD_REQUEST)

        # Get endpoint's total pending requests.
        end_point_id = request_json.get("end_point_id", None)
        pending_requests_num = FEDML_MODEL_CACHE.get_pending_requests_counter(end_point_id)
        if pending_requests_num:
            # Fetch metrics of the past k=3 requests.
            pask_k_metrics = FEDML_MODEL_CACHE.get_endpoint_metrics(
                end_point_id=end_point_id,
                k_recent=3)

            # Get the request timeout from the endpoint settings.
            request_timeout_s = FEDML_MODEL_CACHE.get_endpoint_settings(end_point_id) \
                .get(ServerConstants.INFERENCE_REQUEST_TIMEOUT_KEY, ServerConstants.INFERENCE_REQUEST_TIMEOUT_DEFAULT)

            # Only proceed if the past k metrics collection is not empty.
            if pask_k_metrics:
                # Measure the average latency in seconds(!), hence the 0.001 multiplier.
                past_k_latencies_sec = \
                    [float(j_obj["current_latency"]) * 0.001 for j_obj in pask_k_metrics]
                mean_latency = sum(past_k_latencies_sec) / len(past_k_latencies_sec)

                # If timeout threshold is exceeded then cancel and return time out error.
                should_block = (mean_latency * pending_requests_num) > request_timeout_s
                if should_block:
                    return JSONResponse(
                        {"error": True, "message": "Request timed out."},
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT)

    response = await call_next(request)
    return response


@api.on_event("startup")
async def startup_event():
    configure_logging()


@api.get('/')
async def root():
    return {'message': 'TensorOpera Inference Service!'}


@api.get('/ready')
async def ready():
    return {'message': 'Inference gateway is ready.'}


@api.post('/api/v1/predict')
async def predict(request: Request):
    # Get json data
    input_json = await request.json()
    end_point_id = input_json.get("end_point_id", None)

    # Get header
    header = request.headers

    try:
        response = await _predict(end_point_id, input_json, header)
    except Exception as e:
        response = {"error": True, "message": f"{traceback.format_exc()}"}

    return response


@api.post("/inference/{end_point_id}/completions")
@api.post("/inference/{end_point_id}/chat/completions")
async def predict_openai(end_point_id, request: Request):
    # Get json data
    input_json = await request.json()

    # Get header
    header = request.headers

    # translate request keys
    input_json["end_point_name"] = input_json.get("model", None)

    authorization = request.headers.get("Authorization", None)
    if authorization is not None and authorization.startswith("Bearer "):
        input_json["token"] = authorization.split("Bearer ")[-1].strip()

    try:
        response = await _predict(end_point_id, input_json, header)
    except Exception as e:
        response = {"error": True, "message": f"{traceback.format_exc()}, exception {e}"}

    return response


@api.post('/inference/{end_point_id}')
async def predict_with_end_point_id(end_point_id, request: Request, response: Response):
    # Get json data
    input_json = await request.json()

    # Get header
    header = request.headers

    try:
        inference_response = await _predict(end_point_id, input_json, header)

        if isinstance(inference_response, (Response, StreamingResponse)):
            error_code = inference_response.status_code
        elif isinstance(inference_response, Mapping):
            error_code = inference_response.get("error_code")
        else:
            error_code = response.status_code

        if error_code == status.HTTP_404_NOT_FOUND:
            response.status_code = status.HTTP_404_NOT_FOUND
    except Exception as e:
        inference_response = {"error": True, "message": f"{traceback.format_exc()}"}

    return inference_response


async def _predict(
        end_point_id,
        input_json,
        header=None
) -> Union[MutableMapping[str, Any], Response, StreamingResponse]:
    # Always increase the pending requests counter on a new incoming request.
    FEDML_MODEL_CACHE.update_pending_requests_counter(end_point_id, increase=True)
    inference_response = {}

    try:
        in_end_point_id = end_point_id
        in_end_point_name = input_json.get("end_point_name", None)
        in_model_name = input_json.get("model_name", None)
        in_model_version = input_json.get("model_version", None)
        in_end_point_token = input_json.get("token", None)
        in_return_type = "default"
        if header is not None:
            in_return_type = header.get("Accept", "default")

        if in_model_version is None:
            in_model_version = "*"  # * | latest | specific version

        start_time = time.time_ns()

        # Allow missing end_point_name and model_name in the input parameters.
        if in_model_name is None or in_end_point_name is None:
            ret_endpoint_name, ret_model_name = retrieve_info_by_endpoint_id(in_end_point_id, in_end_point_name)
            if in_model_name is None:
                in_model_name = ret_model_name
            if in_end_point_name is None:
                in_end_point_name = ret_endpoint_name

        # Authenticate request token
        if auth_request_token(in_end_point_id, in_end_point_name, in_model_name, in_end_point_token):
            # Check the endpoint is activated
            if not is_endpoint_activated(in_end_point_id):
                inference_response = {"error": True, "message": "endpoint is not activated."}
                logging_inference_request(input_json, inference_response)
                FEDML_MODEL_CACHE.update_pending_requests_counter(end_point_id, decrease=True)
                return inference_response

            # Found idle inference device
            idle_device, end_point_id, model_id, model_name, model_version, inference_host, inference_output_url = \
                found_idle_inference_device(in_end_point_id, in_end_point_name, in_model_name, in_model_version)
            if idle_device is None or idle_device == "":
                FEDML_MODEL_CACHE.update_pending_requests_counter(end_point_id, decrease=True)
                return {"error": True, "error_code": status.HTTP_404_NOT_FOUND,
                        "message": "can not found active inference worker for this endpoint."}

            # Start timing for model metrics
            model_metrics = FedMLModelMetrics(end_point_id, in_end_point_name,
                                              model_id, in_model_name, model_version,
                                              Settings.model_infer_host,
                                              Settings.redis_addr,
                                              Settings.redis_port,
                                              Settings.redis_password,
                                              version=Settings.version)
            # Setting time to the time before authentication and idle device discovery.
            model_metrics.set_start_time(start_time)

            # Send inference request to idle device
            logging.info("inference url {}.".format(inference_output_url))
            if inference_output_url != "":
                input_list = input_json.get("inputs", input_json)
                stream_flag = input_json.get("stream", False)
                input_list["stream"] = input_list.get("stream", stream_flag)
                output_list = input_json.get("outputs", [])
                inference_response = await send_inference_request(
                    idle_device,
                    end_point_id,
                    inference_output_url,
                    input_list,
                    output_list,
                    inference_type=in_return_type)

            # Calculate model metrics
            try:
                model_metrics.calc_metrics(end_point_id, in_end_point_name,
                                           model_id, model_name, model_version,
                                           inference_output_url, idle_device)
            except Exception as e:
                logging.info("Calculate Inference Metrics Exception: {}".format(traceback.format_exc()))
                pass

            logging_inference_request(input_json, inference_response)
            FEDML_MODEL_CACHE.update_pending_requests_counter(end_point_id, decrease=True)
            return inference_response
        else:
            inference_response = {"error": True, "message": "token is not valid."}
            logging_inference_request(input_json, inference_response)
            FEDML_MODEL_CACHE.update_pending_requests_counter(end_point_id, decrease=True)
            return inference_response

    except Exception as e:
        logging.error("Inference Exception: {}".format(traceback.format_exc()))
        # Need to reduce the pending requests counter in whatever exception that may be raised.
        FEDML_MODEL_CACHE.update_pending_requests_counter(end_point_id, decrease=True)


def retrieve_info_by_endpoint_id(end_point_id, in_end_point_name=None, in_model_name=None,
                                 in_model_version=None, enable_check=False):
    """
    We allow missing end_point_name and model_name in the input parameters.
    return end_point_name, model_name
    """
    redis_key = FEDML_MODEL_CACHE.get_end_point_full_key_by_id(end_point_id)
    if redis_key is not None:
        end_point_name = ""
        model_name = ""
        if in_end_point_name is not None:
            end_point_name = in_end_point_name
            model_name = redis_key[
                         len(f"{FedMLModelCache.FEDML_MODEL_DEPLOYMENT_STATUS_TAG}-{end_point_id}-{in_end_point_name}-"):]
        else:
            # e.g. FEDML_MODEL_DEPLOYMENT_STATUS--1234-dummy_endpoint_name-dummy_model_name
            try:
                end_point_id, end_point_name, model_name = redis_key.split("--")[1].split("-")
            except Exception as e:
                logging.warning(f"Failed to parse redis_key: {redis_key}. Could not retrieve only use end_point_id.")

        if enable_check:
            if end_point_name != in_end_point_name or model_name != in_model_name:
                raise Exception("end_point_name or model_name is not matched.")
    else:
        raise Exception("end_point_id is not found.")

    return end_point_name, model_name


def found_idle_inference_device(end_point_id, end_point_name, in_model_name, in_model_version):
    idle_device = ""
    model_name = ""
    model_id = ""
    inference_host = ""
    inference_output_url = ""
    model_version = ""
    # Found idle device (TODO: optimize the algorithm to search best device for inference)
    payload, idle_device = FEDML_MODEL_CACHE. \
        get_idle_device(end_point_id, end_point_name, in_model_name, in_model_version)
    if payload is not None:
        logging.info("found idle deployment result {}".format(payload))
        deployment_result = payload
        model_name = deployment_result["model_name"]
        model_version = deployment_result["model_version"]
        model_id = deployment_result["model_id"]
        end_point_id = deployment_result["end_point_id"]
        inference_output_url = deployment_result["model_url"]
        url_parsed = urlparse(inference_output_url)
        inference_host = url_parsed.hostname
    else:
        logging.info("not found idle deployment result")

    return idle_device, end_point_id, model_id, model_name, model_version, inference_host, inference_output_url


async def send_inference_request(idle_device, end_point_id, inference_url, input_list, output_list,
                                 inference_type="default", has_public_ip=True):
    request_timeout_sec = FEDML_MODEL_CACHE.get_endpoint_settings(end_point_id) \
        .get("request_timeout_sec", ClientConstants.INFERENCE_REQUEST_TIMEOUT)

    try:
        http_infer_available = os.getenv("FEDML_INFERENCE_HTTP_AVAILABLE", True)
        if not http_infer_available:
            if http_infer_available == "False" or http_infer_available == "false":
                http_infer_available = False

        if http_infer_available:
            response_ok = await FedMLHttpInference.is_inference_ready(
                inference_url,
                timeout=request_timeout_sec)
            if response_ok:
                response_ok, inference_response = await FedMLHttpInference.run_http_inference_with_curl_request(
                    inference_url,
                    input_list,
                    output_list,
                    inference_type=inference_type,
                    timeout=request_timeout_sec)
                logging.info(f"Use http inference. return {response_ok}")
                return inference_response

        response_ok = await FedMLHttpProxyInference.is_inference_ready(
            inference_url,
            timeout=request_timeout_sec)
        if response_ok:
            response_ok, inference_response = await FedMLHttpProxyInference.run_http_proxy_inference_with_request(
                end_point_id,
                inference_url,
                input_list,
                output_list,
                inference_type=inference_type,
                timeout=request_timeout_sec)
            logging.info(f"Use http proxy inference. return {response_ok}")
            return inference_response

        if not has_public_ip:
            agent_config = {"mqtt_config": Settings.mqtt_config}
            mqtt_inference = FedMLMqttInference(
                agent_config=agent_config,
                run_id=end_point_id)
            response_ok = mqtt_inference.run_mqtt_health_check_with_request(
                idle_device,
                end_point_id,
                inference_url,
                timeout=request_timeout_sec)
            inference_response = {"error": True, "message": "Failed to use http, http-proxy and mqtt for inference."}
            if response_ok:
                response_ok, inference_response = mqtt_inference.run_mqtt_inference_with_request(
                    idle_device,
                    end_point_id,
                    inference_url,
                    input_list,
                    output_list,
                    inference_type=inference_type,
                    timeout=request_timeout_sec)

            logging.info(f"Use mqtt inference. return {response_ok}.")
            return inference_response
        return {"error": True, "message": "Failed to use http, http-proxy for inference, no response from replica."}
    except Exception as e:
        inference_response = {"error": True,
                              "message": f"Exception when using http, http-proxy and mqtt "
                                         f"for inference: {traceback.format_exc()}."}
        logging.info("Inference Exception: {}".format(traceback.format_exc()))
        return inference_response


def auth_request_token(end_point_id, end_point_name, model_name, token):
    if token is None:
        return False
    cached_token = FEDML_MODEL_CACHE. \
        get_end_point_token(end_point_id, end_point_name, model_name)
    if cached_token is not None and str(cached_token) == str(token):
        return True
    return False


def is_endpoint_activated(end_point_id):
    if end_point_id is None:
        return False
    activated = FEDML_MODEL_CACHE.get_end_point_activation(end_point_id)
    return activated


def logging_inference_request(request, response):
    if os.getenv("ENABLE_FEDML_INFERENCE_LOG", "False") in ["False", "false", "0", ""]:
        return

    try:
        log_dir = ServerConstants.get_log_file_dir()
        inference_log_file = os.path.join(log_dir, "inference.log")
        with open(inference_log_file, "a") as f:
            f.writelines([f"request: {request}, response: {response}\n"])
    except Exception as ex:
        logging.info(f"failed to log inference request and response to file with exception {ex}")


def configure_logging():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args([])

    setattr(args, "log_file_dir", ServerConstants.get_log_file_dir())
    setattr(args, "run_id", -1)
    setattr(args, "role", "server")
    setattr(args, "using_mlops", True)
    setattr(args, "config_version", fedml.get_env_version())

    runner_info = ServerConstants.get_runner_infos()
    if not (runner_info and "edge_id" in runner_info):
        raise Exception("Inference gateway couldn't be started as edge_id couldn't be parsed from runner_infos.yaml")
    setattr(args, "edge_id", int(runner_info.get("edge_id")))

    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.INFO)
    MLOpsRuntimeLogDaemon.get_instance(args).start_log_processor(log_run_id=args.run_id, log_device_id=args.edge_id,
                                                                 log_source=Settings.server_name,
                                                                 log_file_prefix=Settings.server_name)
    logging.info("start the log processor for inference gateway")


if __name__ == "__main__":
    import uvicorn
    port = 2203
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="info")
