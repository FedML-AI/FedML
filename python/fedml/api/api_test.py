import os.path
import time

import fedml

# Login
error_code, error_msg = fedml.api.fedml_login(version="dev")
if error_code != 0:
    print("API Key is invalid!")
    exit(1)

# Yaml file
cur_dir = os.path.dirname(__file__)
fedml_dir = os.path.dirname(cur_dir)
yaml_file = os.path.join(fedml_dir, "computing", "scheduler", "scheduler_entry", "call_gpu.yaml")

# Match resources
resource_id, error_code, error_msg = fedml.api.match_resources(yaml_file)
if error_code != 0:
    print(f"Failed to match resources. Reason: {error_msg}")
    exit(1)

# Launch job
job_id, error_code, error_msg = fedml.api.launch_job(yaml_file, resource_id=resource_id)
if error_code != 0:
    print(f"Failed to launch job. Reason: {error_msg}")
    exit(1)

# Get job status
job_status, total_num, total_pages, job_logs = fedml.api.launch_log(job_id, 1, 100)
if job_status is None:
    print(f"Failed to get job status. Reason: {error_msg}")
    exit(1)
print(f"Job status {job_status}")

# Get job logs
time.sleep(30)
job_status, total_num, total_pages, job_logs = fedml.api.launch_log(job_id, 1, 100)
if job_status is None:
    print(f"Failed to get job logs. Reason: {error_msg}")
    exit(1)
print(f"Job logs {job_logs}")

