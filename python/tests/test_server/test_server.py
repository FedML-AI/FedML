import os.path
import time
import fedml
from fedml.api.constants import RunStatus

# Login
fedml.set_env_version("test")
fedml.set_local_on_premise_platform_port(18080)
error_code, error_msg = fedml.api.fedml_login(api_key="")
if error_code != 0:
    raise Exception("API Key is invalid!")

# Yaml file
cur_dir = os.path.dirname(__file__)
fedml_dir = os.path.dirname(cur_dir)
python_dir = os.path.dirname(fedml_dir)
yaml_file = os.path.join(python_dir, "examples", "launch", "serve_job_mnist.yaml")

# Launch job
launch_result_dict = {}
launch_result_status = {}

launch_result = fedml.api.launch_job(yaml_file)

# launch_result = fedml.api.launch_job_on_cluster(yaml_file, "alex-cluster")
if launch_result.result_code != 0:
    raise Exception(f"Failed to launch job. Reason: {launch_result.result_message}")

launch_result_dict[launch_result.run_id] = launch_result
launch_result_status[launch_result.run_id] = RunStatus.STARTING
