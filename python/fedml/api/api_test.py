import os.path

import fedml

fedml.api.fedml_login(version="dev")

cur_dir = os.path.dirname(__file__)
fedml_dir = os.path.dirname(cur_dir)
yaml_file = os.path.join(fedml_dir, "computing", "scheduler", "scheduler_entry", "call_gpu.yaml")
job_id, err_code, err_msg = fedml.api.launch_job(yaml_file)

job_status, total_num, total_pages, job_logs = fedml.api.launch_log(job_id, 1, 100)
print(f"Job status {job_status}")
print(f"Job logs {job_logs}")

