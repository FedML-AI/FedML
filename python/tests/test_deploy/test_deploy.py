import os.path
import time
import fedml
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
print("Endpoint id is", launch_result.inner_id)

cnt = 0
while 1:
    try:
        r = fedml.api.get_endpoint(endpoint_id=launch_result.inner_id)
    except Exception as e:
        raise Exception(f"FAILED to get endpoint:{launch_result.inner_id}. {e}")
    if r.status == "DEPLOYED":
        print("Deployment has been successfully!")
        break 
    elif r.status == "FAILED":
        raise Exception("FAILED to deploy.")
    time.sleep(1)
    cnt += 1
    if cnt %3 ==0:
        print('Deployment status is', r.status)