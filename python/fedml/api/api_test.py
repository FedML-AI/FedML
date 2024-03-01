import os.path
import time

import fedml

# Login
fedml.set_env_version("local")
fedml.set_local_on_premise_platform_port(18080)
error_code, error_msg = fedml.api.fedml_login(api_key="1316b93c82da40ce90113a2ed12f0b14")
if error_code != 0:
    print("API Key is invalid!")
    exit(1)

# Yaml file
cur_dir = os.path.dirname(__file__)
fedml_dir = os.path.dirname(cur_dir)
python_dir = os.path.dirname(fedml_dir)
yaml_file = os.path.join(python_dir, "examples", "launch", "hello_job.yaml")

# Launch job
for i in range(0, 10):
    launch_result = fedml.api.launch_job(yaml_file)
    # launch_result = fedml.api.launch_job_on_cluster(yaml_file, "alex-cluster")
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

