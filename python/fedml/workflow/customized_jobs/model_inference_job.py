
from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.model_scheduler.device_model_object import FedMLEndpointDetail
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.workflow.jobs import JobStatus
import requests


class ModelInferenceJob(CustomizedBaseJob):
    def __init__(self, name, endpoint_id=None, job_api_key=None):
        super().__init__(name, job_api_key=job_api_key)
        self.endpoint_id = endpoint_id
        self.endpoint_detail: FedMLEndpointDetail = None
        self.inference_url = None
        self.infer_request_body = None
        self.key_token = job_api_key
        self.out_response_json = None

        try:
            self.endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(self.endpoint_id, self.job_api_key)
        except Exception as e:
            self.endpoint_detail = None

        self.run_status = JobStatus.PROVISIONING

    def run(self):
        if self.endpoint_detail is None:
            try:
                self.endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(self.endpoint_id, self.job_api_key)
            except Exception as e:
                self.endpoint_detail = None

        if self.endpoint_detail is None:
            self.run_status = JobStatus.FAILED

            self.output_data_dict = {
                "error": True, "message": f"Can't get the detail info of endpoint {self.endpoint_id}."}

            return

        if self.endpoint_detail.status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
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

        response = requests.post(self.inference_url, headers=headers, json=self.infer_request_body)
        if response.status_code != 200:
            print(f"Inference response with status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            return {"error": True, "message": response.content()}
        else:
            return response.json()


