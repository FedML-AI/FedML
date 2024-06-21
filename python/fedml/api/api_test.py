import os.path
import time

import fedml

# Login
fedml.set_env_version("test")
fedml.set_local_on_premise_platform_port(18080)
error_code, error_msg = fedml.api.fedml_login(api_key="")
if error_code != 0:
    print("API Key is invalid!")
    exit(1)

# Yaml file
cur_dir = os.path.dirname(__file__)
fedml_dir = os.path.dirname(cur_dir)
python_dir = os.path.dirname(fedml_dir)
yaml_file = os.path.join(python_dir, "examples", "launch", "hello_job.yaml")

# Launch job
launch_result_list = list()
for i in range(0, 10):
    launch_result = fedml.api.launch_job(yaml_file)
    launch_result_list.append(launch_result)
    # launch_result = fedml.api.launch_job_on_cluster(yaml_file, "alex-cluster")
    if launch_result.result_code != 0:
        print(f"Failed to launch job. Reason: {launch_result.result_message}")

# Get job status
while len(launch_result_list) > 0:
    for launch_result in launch_result_list:
        log_result = fedml.api.run_logs(launch_result.run_id, 1, 5)
        if log_result is None or log_result.run_status is None:
            print(f"Failed to get job status.")
            #exit(1)
        print(f"Run {launch_result.run_id}, status {log_result.run_status}")
        time.sleep(0.5)

# Get job logs
time.sleep(30)
log_result = fedml.api.run_logs(launch_result.run_id, 1, 100)
if log_result is None or log_result.run_status is None:
    print(f"Failed to get run logs.")
    exit(1)
print(f"Run logs {log_result.log_line_list}")

