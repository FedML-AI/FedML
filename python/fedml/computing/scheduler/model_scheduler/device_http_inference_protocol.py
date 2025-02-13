import logging
import time
import uuid

import httpx
import traceback

from .device_client_constants import ClientConstants

from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from urllib.parse import urlparse
from typing import Mapping


class FedMLHttpInference:
    _http_client = None  # Class variable for shared HTTP client

    @classmethod
    async def get_http_client(cls):
        if cls._http_client is None:
            limits = httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000,
                keepalive_expiry=60
            )
            cls._http_client = httpx.AsyncClient(limits=limits)
        return cls._http_client

    def __init__(self):
        pass

    @staticmethod    
    async def is_inference_ready(inference_url, path="ready", timeout=None):
        """
        True: inference is ready
        False: cannot be reached, will try other protocols
        None: can be reached, but not ready
        """
        url_parsed = urlparse(inference_url)
        ready_url = f"http://{url_parsed.hostname}:{url_parsed.port}/{path}"
        response_ok = False

        # TODO (Raphael): Support more methods and return codes rules.
        try:
            # async with httpx.AsyncClient() as client:
            client = await FedMLHttpInference.get_http_client()
            ready_response = await client.get(url=ready_url, timeout=timeout)

            if isinstance(ready_response, (Response, StreamingResponse)):
                error_code = ready_response.status_code
            elif isinstance(ready_response, Mapping):
                error_code = ready_response.get("error_code")
            else:
                error_code = ready_response.status_code

            if error_code == 200:
                response_ok = True
            else:
                response_ok = None
        except Exception as e:
            response_ok = False

        return response_ok

    @staticmethod
    async def run_http_inference_with_curl_request(
            inference_url, inference_input_list, inference_output_list,
            inference_type="default", engine_type="default", timeout=None, method="POST"
    ):
        if inference_type == "default":
            model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                                 'Accept': 'application/json'}
        else:
            model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                                 'Accept': inference_type}
        if engine_type == "default":
            model_inference_json = inference_input_list
        else:  # triton
            model_inference_json = {
                "inputs": inference_input_list,
                "outputs": inference_output_list
            }

        try:
            if model_inference_json.get("stream", False):
                model_inference_result = StreamingResponse(
                    stream_generator(inference_url, input_json=model_inference_json, method=method),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": model_api_headers.get("Accept", "text/event-stream"),
                        "Cache-Control": "no-cache",
                    }
                )
                response_ok = True
            else:
                response_ok, model_inference_result = await redirect_non_stream_req_to_worker(
                    inference_type, inference_url, model_api_headers, model_inference_json, timeout, method=method)
        except Exception as e:
            response_ok = False
            model_inference_result = {"response": f"{traceback.format_exc()}"}

        return response_ok, model_inference_result


async def stream_generator(inference_url, input_json, method="POST"):
    # async with httpx.AsyncClient() as client:
    client = await FedMLHttpInference.get_http_client()
    async with client.stream(method, inference_url, json=input_json,
                                timeout=ClientConstants.WORKER_STREAM_API_TIMEOUT) as response:
        async for chunk in response.aiter_lines():
            # we consumed a newline, need to put it back
            yield f"{chunk}\n"


async def redirect_non_stream_req_to_worker(inference_type, inference_url, model_api_headers, model_inference_json,
                                            timeout=None, method="POST"):
    response_ok = True
    # request_id = str(uuid.uuid4())[:8]
    # start_time = time.time()
    # logging.info(f"[Request-{request_id}] Starting HTTP request to {inference_url}")
    
    try:
         # async with httpx.AsyncClient() as client:
        client = await FedMLHttpInference.get_http_client()
        response = await client.request(
            method=method, url=inference_url, headers=model_api_headers, json=model_inference_json, timeout=timeout
        )
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # logging.info(f"[Request-{request_id}] Completed HTTP request. Time taken: {elapsed_time:.3f} seconds")
    except Exception as e:
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # logging.error(f"[Request-{request_id}] Failed HTTP request after {elapsed_time:.3f} seconds. Error: {str(e)}")
        response_ok = False
        model_inference_result = {"error": e}
        return response_ok, model_inference_result
    
    if response.status_code == 200:
        try:
            if inference_type == "image/png":
                # wrapped media type for image
                binary_content: bytes = response.content
                model_inference_result = Response(content=binary_content, media_type=inference_type)
            else:
                model_inference_result = response.json()
        except Exception as e:
            response_ok = True
            logging.warning(f"Status code 200, but cannot trans response to json due to: {e}.")
            model_inference_result = {"response": f"{response.content}"}
    else:
        model_inference_result = {"response": f"{response.content}"}

    return response_ok, model_inference_result
    