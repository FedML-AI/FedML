import os
import traceback
from typing import Mapping
from urllib.parse import urlparse
from .device_client_constants import ClientConstants
from .device_server_constants import ServerConstants
import requests
import httpx
from fastapi.responses import Response
from fastapi.responses import StreamingResponse


class FedMLHttpProxyInference:
    def __init__(self):
        pass

    @staticmethod
    async def is_inference_ready(proxy_url, inference_url, timeout=None) -> bool:
        http_proxy_url = f"{proxy_url}/ready"
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                             'Accept': 'application/json'}
        print(f"Check if the proxy {http_proxy_url} is ready for container address {inference_url}")
        model_ready_json = {
            "inference_url": inference_url,
        }
        if timeout is not None:
            model_ready_json["inference_timeout"] = timeout

        response_ok = False
        try:
            async with httpx.AsyncClient() as client:
                ready_response = await client.post(
                    url=http_proxy_url, headers=model_api_headers, json=model_ready_json, timeout=timeout
                )

            if isinstance(ready_response, (Response, StreamingResponse)):
                error_code = ready_response.status_code
            elif isinstance(ready_response, Mapping):
                error_code = ready_response.get("error_code")
            else:
                error_code = ready_response.status_code

            if error_code == 200:
                response_ok = True
        except Exception as e:
            response_ok = False

        return response_ok

    @staticmethod
    async def run_http_proxy_inference_with_request(
            endpoint_id, inference_url, inference_input_list,
            inference_output_list, inference_type="default",
            timeout=None, inference_proxy_port=ClientConstants.WORKER_PROXY_PORT_EXTERNAL
    ):
        inference_response = {}
        http_proxy_url = f"http://{urlparse(inference_url).hostname}:{inference_proxy_port}/api/v1/predict"
        if inference_type == "default":
            model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                                 'Accept': 'application/json'}
        else:
            model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                                 'Accept': inference_type}
        model_inference_json = {
            "endpoint_id": endpoint_id,
            "inference_url": inference_url,
            "input": inference_input_list,
            "output": inference_output_list,
            "inference_type": inference_type
        }
        if timeout is not None:
            model_inference_json["inference_timeout"] = timeout

        response_ok = False
        model_inference_result = {}
        try:
            async with httpx.AsyncClient() as client:
                inference_response = await client.post(
                    url=http_proxy_url, headers=model_api_headers, json=model_inference_json, timeout=timeout
                )

            if isinstance(inference_response, (Response, StreamingResponse)):
                error_code = inference_response.status_code
            elif isinstance(inference_response, Mapping):
                error_code = inference_response.get("error_code")
            else:
                error_code = inference_response.status_code

            if error_code == 200:
                response_ok = True
                return response_ok, inference_response.json()
            else:
                model_inference_result = {"response": f"{inference_response.content}"}
        except Exception as e:
            response_ok = False
            model_inference_result = {"response": f"{traceback.format_exc()}"}

        return response_ok, model_inference_result

    @staticmethod
    def allocate_worker_proxy_port(internal_port_in_yaml=None, external_port_in_yaml=None) -> (str, str):
        """
        Return: (worker_proxy_internal_port, worker_proxy_external_port)
        Priority: yaml > env > default
        """
        ret_port_internal, ret_port_external = \
            ClientConstants.WORKER_PROXY_PORT_INTERNAL, ClientConstants.WORKER_PROXY_PORT_EXTERNAL

        if os.getenv("FEDML_WORKER_PORT_INTERNAL", None):
            ret_port_internal = os.getenv("FEDML_WORKER_PORT_INTERNAL")
        if os.getenv("FEDML_WORKER_PORT_EXTERNAL", None):
            ret_port_external = os.getenv("FEDML_WORKER_PORT_EXTERNAL")

        if internal_port_in_yaml:
            ret_port_internal = internal_port_in_yaml
        if external_port_in_yaml:
            ret_port_external = external_port_in_yaml

        return ret_port_internal, ret_port_external

    @staticmethod
    def allocate_master_proxy_port(internal_port_in_yaml=None, external_port_in_yaml=None) -> (str, str):
        """
        Return: (master_proxy_internal_port, master_proxy_external_port)
        Priority: yaml > env > default
        """
        ret_port_internal, ret_port_external = \
            ServerConstants.MASTER_PROXY_PORT_INTERNAL, ServerConstants.MASTER_PROXY_PORT_EXTERNAL

        if os.getenv("FEDML_MASTER_PORT_INTERNAL", None):
            ret_port_external = os.getenv("FEDML_MASTER_PORT_INTERNAL")
        if os.getenv("FEDML_MASTER_PORT_EXTERNAL", None):
            ret_port_external = os.getenv("FEDML_MASTER_PORT_EXTERNAL")

        if internal_port_in_yaml:
            ret_port_internal = internal_port_in_yaml
        if external_port_in_yaml:
            ret_port_external = external_port_in_yaml

        return ret_port_internal, ret_port_external
