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
yaml_file = os.path.join(python_dir, "examples", "launch", "hello_job.yaml")

# Launch job
launch_result_dict = {}
launch_result_status = {}

for i in range(0, 10):
    launch_result = fedml.api.launch_job(yaml_file)
    
    # launch_result = fedml.api.launch_job_on_cluster(yaml_file, "alex-cluster")
    if launch_result.result_code != 0:
        raise Exception(f"Failed to launch job. Reason: {launch_result.result_message}")

    launch_result_dict[launch_result.run_id] = launch_result
    launch_result_status[launch_result.run_id] = RunStatus.STARTING

def check_status(status_dict):

    all_success = True
    for key, value in status_dict.items():
        if value not in [RunStatus.FINISHED]:
            all_success = False
            break
    return all_success 
        
# check job status
while 1:
    time.sleep(5)
    check_all = check_status(launch_result_status)
    if check_all == True:
        print("Check that all tasks have run successfully!")
        break

    for run_id, launch_result in launch_result_dict.items():
        if launch_result_status[run_id] == RunStatus.FINISHED:
            continue
        
        log_result = fedml.api.run_logs(launch_result.run_id, 1, 5)
        if log_result is None or log_result.run_status is None:
            raise Exception(f"Failed to get job status.")

        print(f"run_id: {launch_result.run_id} run_status: {log_result.run_status}")
        launch_result_status[launch_result.run_id] = log_result.run_status
        if log_result.run_status in [RunStatus.ERROR, RunStatus.FAILED]:
            log_result = fedml.api.run_logs(launch_result.run_id, 1, 100)
            if log_result is None or log_result.run_status is None:
                raise Exception(f"run_id:{run_id} run_status:{log_result.run_status} and failed to get run logs.")

            raise Exception(f"run_id:{run_id} run_status:{log_result.run_status} run logs: {log_result.log_line_list}")

           

