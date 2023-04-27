from .fedml_client import FedMLModelServingClient as Client
from .fedml_server import FedMLModelServingServer as Server

__all__ = [
    "Client",
    "Server",
]
