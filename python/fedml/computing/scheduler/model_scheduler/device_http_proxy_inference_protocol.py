import traceback
from typing import Mapping
from urllib.parse import urlparse
from .device_client_constants import ClientConstants
import requests
import httpx
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from .device_http_inference_protocol import stream_generator


class FedMLHttpProxyInference:
    def __init__(self):
        pass

    @staticmethod
    async def is_inference_ready(inference_url, timeout=None) -> bool:
        http_proxy_url = f"http://{urlparse(inference_url).hostname}:{ClientConstants.LOCAL_CLIENT_API_PORT}/ready"
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                             'Accept': 'application/json'}
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
            if inference_input_list.get("stream", False):
                model_inference_result = StreamingResponse(
                    stream_generator(http_proxy_url, input_json=model_inference_json),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": model_api_headers.get("Accept", "text/event-stream"),
                        "Cache-Control": "no-cache",
                    }
                )
                response_ok = True
            else:
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
                    if inference_type == "default":
                        model_inference_result = inference_response.json()
                    elif inference_type == "image/png":
                        binary_content: bytes = inference_response.content
                        model_inference_result = Response(content=binary_content, media_type="image/png")
                    else:
                        model_inference_result = inference_response.json()
                    response_ok = True
                else:
                    model_inference_result = {"response": f"{inference_response.content}"}
        except Exception as e:
            response_ok = False
            model_inference_result = {"response": f"{traceback.format_exc()}"}

        return response_ok, model_inference_result
