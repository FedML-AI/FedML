import json
import logging
import multiprocessing
import sys

from multiprocessing import Process
import os
import platform
import shutil
import subprocess
import threading

import time
import uuid

import fedml
from fedml import mlops
from ..comm_utils import sys_utils

from ....core.distributed.communication.mqtt.mqtt_manager import MqttManager

from ....core.mlops.mlops_metrics import MLOpsMetrics
from urllib.parse import urlparse
from .device_client_constants import ClientConstants
import requests


class FedMLHttpProxyInfernce:
    def __init__(self):
        pass

    def run_http_proxy_inference_with_request(
            self, endpoint_id, inference_url, inference_input_list,
            inference_output_list, inference_type="default"
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

        response_ok = False
        try:
            inference_response = requests.post(inference_url, headers=model_api_headers, json=model_inference_json)
            if inference_response.status_code == 200:
                response_ok = True
                return response_ok, inference_response
        except Exception as e:
            print("Error in running inference: {}".format(e))

        return response_ok, inference_response
