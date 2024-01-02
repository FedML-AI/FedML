import fedml
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants

# Set Your Dev -> "dev" | "release"
fedml.set_env_version("dev")

print(ClientConstants.get_model_ops_url())