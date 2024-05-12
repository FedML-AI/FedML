
from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.model_scheduler.device_model_object import FedMLEndpointDetail
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.workflow.jobs import JobStatus
import requests
from fedml.workflow.workflow_mlops_api import WorkflowMLOpsApi


class ModelInferenceJob(CustomizedBaseJob):
    def __init__(self, name, endpoint_name=None, job_api_key=None, endpoint_user_name=None):
        super().__init__(name, job_api_key=job_api_key)
        self.endpoint_id = None
        self.endpoint_name = endpoint_name
        self.endpoint_user_name = endpoint_user_name
        self.endpoint_detail: FedMLEndpointDetail = None
        self.inference_url = None
        self.infer_request_body = None
        self.key_token = job_api_key
        self.out_response_json = None

        try:
            self.endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(
                endpoint_name=self.endpoint_name, user_api_key=self.job_api_key)
        except Exception as e:
            self.endpoint_detail = None

        self.run_status = JobStatus.PROVISIONING

    def run(self):
        if self.endpoint_detail is None:
            try:
                self.endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(
                    endpoint_name=self.endpoint_name, user_api_key=self.job_api_key)
            except Exception as e:
                self.endpoint_detail = None

        if self.endpoint_detail is None:
            self.run_status = JobStatus.FAILED

            self.output_data_dict = {
                "error": True, "message": f"Can't get the detail info of endpoint {self.endpoint_id}."}

            return

        if self.endpoint_detail.status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            print("Predicting..., please wait.")
            self.run_status = JobStatus.RUNNING

            self._build_in_params()

            self.out_response_json = self._inference()

            self.run_status = JobStatus.FAILED if self.out_response_json.get("error", False) else JobStatus.FINISHED

            self.output_data_dict = self.out_response_json
        else:
            self.run_status = JobStatus.FAILED

            self.output_data_list = {
                "error": True,
                "message": f"The endpoint {self.endpoint_id} is in the status {self.endpoint_detail.status}."}

    def status(self):
        return self.run_status

    def kill(self):
        pass

    def _build_in_params(self):
        self.inference_url = self.endpoint_detail.inference_url

        if self.input_data_dict is not None and isinstance(self.input_data_dict, dict):
            self.infer_request_body = self.input_data_dict
        else:
            self.infer_request_body = self.endpoint_detail.input_json

    def _inference(self):
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key_token}',
        }

        model_name = self.infer_request_body.get("model")
        if model_name is None:
            self.infer_request_body["model"] = f"{self.endpoint_user_name}/{self.endpoint_name}"

        response = requests.post(self.inference_url, headers=headers, json=self.infer_request_body, timeout=60*10)
        if response.status_code != 200:
            print(f"Inference response with status_code = {response.status_code}, "
                  f"response.content: {str(response.content)}")
            return {"error": True, "message": str(response.content)}
        else:
            return response.json()

    def update_run_metadata(self):
        if self.endpoint_detail is None:
            try:
                self.endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(
                    endpoint_name=self.endpoint_name, user_api_key=self.job_api_key)
            except Exception as e:
                self.endpoint_detail = None

        if self.endpoint_detail is None:
            raise Exception("Unable to query endpoint details from the backend.")

        dependency_list = list()
        for dep in self.dependencies:
            dependency_list.append(dep)
        result = WorkflowMLOpsApi.add_run(
            workflow_id=self.workflow_id, job_name=self.name, run_id=self.endpoint_detail.endpoint_id,
            dependencies=dependency_list, api_key=self.job_api_key
        )
        if not result:
            raise Exception(f"Unable to add endpoint {self.endpoint_detail.endpoint_name} "
                            f"to the workflow in the backend.")



