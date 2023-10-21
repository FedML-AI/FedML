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
job_id, project_id, inner_id, error_code, error_msg = fedml.api.launch_job(yaml_file)
if error_code != 0:
    print(f"Failed to launch job. Reason: {error_msg}")
    exit(1)

# Get job status
run_status, total_num, total_pages, log_line_list, job_log_obj = fedml.api.run_logs(job_id, 1, 100)
if run_status is None:
    print(f"Failed to get job status. Reason: {error_msg}")
    exit(1)
print(f"Run status {run_status}")

# Get job logs
time.sleep(30)
run_status, total_num, total_pages, log_line_list, job_log_obj = fedml.api.run_logs(job_id, 1, 100)
if run_status is None:
    print(f"Failed to get run logs. Reason: {error_msg}")
    exit(1)
print(f"Run logs {log_line_list}")

