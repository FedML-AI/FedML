from .fedml_client import FedMLInferenceClient as Client
from .fedml_server import FedMLInferenceServer as Server

__all__ = [
    "Client",
    "Server",
]
