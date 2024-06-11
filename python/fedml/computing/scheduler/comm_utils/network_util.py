import os
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants


def return_this_device_connectivity_type() -> str:
    """
    Return -> "http" | "http_proxy" |"mqtt"
    """
    if os.environ.get(ClientConstants.ENV_CONNECTION_TYPE_KEY) == ClientConstants.WORKER_CONNECTIVITY_TYPE_HTTP:
        return ClientConstants.WORKER_CONNECTIVITY_TYPE_HTTP
    elif os.environ.get(ClientConstants.ENV_CONNECTION_TYPE_KEY) == ClientConstants.WORKER_CONNECTIVITY_TYPE_HTTP_PROXY:
        return ClientConstants.WORKER_CONNECTIVITY_TYPE_HTTP_PROXY
    elif os.environ.get(ClientConstants.ENV_CONNECTION_TYPE_KEY) == ClientConstants.WORKER_CONNECTIVITY_TYPE_MQTT:
        return ClientConstants.WORKER_CONNECTIVITY_TYPE_MQTT
    else:
        return ClientConstants.WORKER_CONNECTIVITY_TYPE_HTTP
