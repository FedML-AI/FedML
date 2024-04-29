import fedml
import requests
from enum import IntEnum, Enum
from typing import List
from fedml.core.mlops import MLOpsConfigs


class WorkflowType(IntEnum):
    WORKFLOW_TYPE_TRAIN = 0
    WORKFLOW_TYPE_DEPLOY = 1


class WorkflowStatus(Enum):
    PROVISIONING = 0, "PROVISIONING"
    RUNNING = 1, "RUNNING"
    FINISHED = 2, "FINISHED"
    FAILED = 3, "FAILED"
    UNDETERMINED = 4, "UNDETERMINED"

    def __int__(self):
        return self.value[0]

    def __str__(self):
        return self.value[1]


class WorkflowMLOpsApi:

    def __init__(self):
        pass

    @staticmethod
    def create_workflow(workflow_name: str, workflow_type: WorkflowType, api_key: str = None) -> int:
        request_url = WorkflowMLOpsApi.get_mlops_workflow_create_url()
        api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                       "Authorization": f"Bearer {api_key}"}
        request_body = {"name": workflow_name, "type": workflow_type.value}
        _, cert_path = MLOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    request_url, verify=True, headers=api_headers, json=request_body
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    request_url, verify=True, headers=api_headers, json=request_body
                )
        else:
            response = requests.post(request_url, headers=api_headers, json=request_body)
        if response.status_code != 200:
            print(f"Response failed: {response.content}")
            return None
        else:
            resp_data = response.json()
            if resp_data["code"] != "SUCCESS":
                print("Error: {}.".format(resp_data["message"]))
                return None
            message_data = resp_data.get("data", None)
            if message_data is None:
                print(f"Workflow id is empty.")
                return None
            workflow_id = message_data

            return workflow_id

    @staticmethod
    def add_run(workflow_id: int, job_name: str, run_id: int,
                dependencies: List[str] = None, api_key: str = None) -> bool:
        request_url = WorkflowMLOpsApi.get_mlops_workflow_add_run_url()
        api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                       "Authorization": f"Bearer {api_key}"}
        request_body = {
            "jobName": job_name,
            "workflowId": workflow_id
        }
        if run_id is not None:
            request_body["runId"] = run_id
        if dependencies is not None and len(dependencies) > 0:
            request_body["dependencies"] = list()
            for dependency in dependencies:
                request_body["dependencies"].append(
                    {"jobName": dependency})
        _, cert_path = MLOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    request_url, verify=True, headers=api_headers, json=request_body
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    request_url, verify=True, headers=api_headers, json=request_body
                )
        else:
            response = requests.post(request_url, headers=api_headers, json=request_body)
        if response.status_code != 200:
            print(f"Response failed: {response.content}")
            return False
        else:
            resp_data = response.json()
            if resp_data["code"] != "SUCCESS":
                print("Error: {}.".format(resp_data["message"]))
                return False
            message_data = resp_data.get("data", None)
            if message_data is None:
                print(f"Workflow id is empty.")
                return False
            ret_ok = message_data

            return ret_ok

    @staticmethod
    def update_workflow(workflow_id: int, workflow_status: WorkflowStatus, api_key: str = None) -> int:
        request_url = WorkflowMLOpsApi.get_mlops_workflow_create_url()
        api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                       "Authorization": f"Bearer {api_key}"}
        request_body = {"workflowId": workflow_id, "status": int(workflow_status)}
        _, cert_path = MLOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.put(
                    request_url, verify=True, headers=api_headers, json=request_body
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.put(
                    request_url, verify=True, headers=api_headers, json=request_body
                )
        else:
            response = requests.put(request_url, headers=api_headers, json=request_body)
        if response.status_code != 200:
            print(f"Response failed: {response.content}")
            return False
        else:
            resp_data = response.json()
            if resp_data["code"] != "SUCCESS":
                print("Error: {}.".format(resp_data["message"]))
                return False
            message_data = resp_data.get("data", None)
            if message_data is None:
                print(f"Workflow id is empty.")
                return False
            result = message_data

            return result

    @staticmethod
    def get_mlops_workflow_create_url():
        ret_url = f"{fedml._get_backend_service()}/cheetah/cli/workflow"
        return ret_url

    @staticmethod
    def get_mlops_workflow_add_run_url():
        ret_url = f"{fedml._get_backend_service()}/cheetah/cli/workflow/run"
        return ret_url


if __name__ == "__main__":
    test_api_key = "1316b93c82da40ce90113a2ed12f0b14"
    fedml.set_env_version("test")

    workflow_id = WorkflowMLOpsApi.create_workflow(
        workflow_name="test_flow1", workflow_type=WorkflowType.WORKFLOW_TYPE_TRAIN,
        api_key=test_api_key)
    print(f"Workflow id from api: {workflow_id}")

    result = WorkflowMLOpsApi.add_run(
        workflow_id=workflow_id, job_name="test_job2", run_id=1123333,
        dependencies=["test_job1"], api_key=test_api_key
    )
    print(f"Result for adding run to workflow: {result}")

