import os.path
import time

import fedml

# Login
fedml.set_env_version("dev")
error_code, error_msg = fedml.api.fedml_login()
if error_code != 0:
    print("API Key is invalid!")
    exit(1)

# Yaml file
cur_dir = os.path.dirname(__file__)
fedml_dir = os.path.dirname(cur_dir)
python_dir = os.path.dirname(fedml_dir)
yaml_file = os.path.join(python_dir, "examples", "launch", "hello_job.yaml")

# Launch job
launch_result = fedml.api.launch_job(yaml_file)
if launch_result.result_code != 0:
    print(f"Failed to launch job. Reason: {launch_result.result_message}")
    exit(1)

# Get job status
log_result = fedml.api.run_logs(launch_result.run_id, 1, 100)
if log_result is None or log_result.run_status is None:
    print(f"Failed to get job status.")
    exit(1)
print(f"Run status {log_result.run_status}")

# Get job logs
time.sleep(30)
log_result = fedml.api.run_logs(launch_result.run_id, 1, 100)
if log_result is None or log_result.run_status is None:
    print(f"Failed to get run logs.")
    exit(1)
print(f"Run logs {log_result.log_line_list}")

