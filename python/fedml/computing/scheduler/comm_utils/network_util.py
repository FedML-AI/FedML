import os
from urllib.parse import urlparse
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


def replace_url_with_path(url: str, path: str) -> str:
    """
    Replace the path of the URL with the given path.
    """
    if path is None:
        return url
    url_parsed = urlparse(url)
    return f"{url_parsed.scheme}://{url_parsed.netloc}/{path}"
