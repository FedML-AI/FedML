import traceback
from urllib.parse import urlparse
from .device_client_constants import ClientConstants
import requests


class FedMLHttpProxyInference:
    def __init__(self):
        pass

    @staticmethod
    def run_http_proxy_inference_with_request(
            endpoint_id, inference_url, inference_input_list,
            inference_output_list, inference_type="default",
            timeout=None
    ):
        inference_response = {}
        http_proxy_url = f"http://{urlparse(inference_url).hostname}:{ClientConstants.LOCAL_CLIENT_API_PORT}/api/v1/predict"
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
            if timeout is None:
                inference_response = requests.post(http_proxy_url, headers=model_api_headers, json=model_inference_json)
            else:
                inference_response = requests.post(
                    http_proxy_url, headers=model_api_headers, json=model_inference_json, timeout=timeout)
            if inference_response.status_code == 200:
                response_ok = True
                return response_ok, inference_response.content
            else:
                model_inference_result = {"response": f"{inference_response.content}"}
        except Exception as e:
            response_ok = False
            model_inference_result = {"response": f"{traceback.format_exc()}"}

        return response_ok, model_inference_result
