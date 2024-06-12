import os
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants


def return_this_device_connectivity_type() -> str:
    """
    Return -> "http" | "http_proxy" |"mqtt"
    """
    # Get the environmental variable's value and convert to lower case.
    env_conn_type = os.getenv(ClientConstants.ENV_CONNECTION_TYPE_KEY, "").lower()
    if env_conn_type in [
        ClientConstants.WORKER_CONNECTIVITY_TYPE_HTTP,
        ClientConstants.WORKER_CONNECTIVITY_TYPE_HTTP_PROXY,
        ClientConstants.WORKER_CONNECTIVITY_TYPE_MQTT
    ]:
        return env_conn_type
    else:
        return ClientConstants.WORKER_CONNECTIVITY_TYPE_DEFAULT
