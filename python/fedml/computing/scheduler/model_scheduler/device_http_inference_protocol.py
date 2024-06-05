import httpx
import traceback

from .device_client_constants import ClientConstants

from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from urllib.parse import urlparse
from typing import Mapping


class FedMLHttpInference:
    def __init__(self):
        pass

    @staticmethod    
    async def is_inference_ready(inference_url, timeout=None):
        '''
        True: inference is ready
        False: cannot be reached, will try other protocols
        None: can be reached, but not ready
        '''
        url_parsed = urlparse(inference_url)
        ready_url = f"http://{url_parsed.hostname}:{url_parsed.port}/ready"
        response_ok = False
        try:
            async with httpx.AsyncClient() as client:
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
            inference_type="default", engine_type="default", timeout=None
    ):
        model_inference_result = {}
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

        response_ok = False
        try:
            if model_inference_json.get("stream", False):
                model_inference_result = StreamingResponse(
                    stream_generator(inference_url, input_json=model_inference_json),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": model_api_headers.get("Accept", "text/event-stream"),
                        "Cache-Control": "no-cache",
                    }
                )
                response_ok = True
            else:
                response_ok, model_inference_result = await redirect_request_to_worker(
                    inference_type, inference_url, model_api_headers, model_inference_json, timeout)
        except Exception as e:
            response_ok = False
            model_inference_result = {"response": f"{traceback.format_exc()}"}

        return response_ok, model_inference_result


async def stream_generator(inference_url, input_json):
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", inference_url, json=input_json,
                                 timeout=ClientConstants.WORKER_STREAM_API_TIMEOUT) as response:
            async for chunk in response.aiter_lines():
                # we consumed a newline, need to put it back
                yield f"{chunk}\n"


async def redirect_request_to_worker(inference_type, inference_url, model_api_headers, model_inference_json, timeout=None):
    response_ok = True
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=inference_url, headers=model_api_headers, json=model_inference_json, timeout=timeout
            )
    except Exception as e:
        response_ok = False
        model_inference_result = {"error": e}
        return response_ok, model_inference_result
    
    if response.status_code == 200:
        if inference_type == "default":
            model_inference_result = response.json()
        elif inference_type == "image/png":
            binary_content: bytes = response.content
            model_inference_result = Response(content=binary_content, media_type="image/png")
        else:
            model_inference_result = response.json()
    else:
        model_inference_result = {"response": f"{response.content}"}

    return response_ok, model_inference_result
    