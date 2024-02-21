
from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.workflow.jobs import JobStatus
import time
import fedml


class ModelDeployJob(CustomizedBaseJob):
    ALLOWED_MAX_RUNNINT_TIME = 2 * 60 * 60

    def __init__(self, name, endpoint_id=None, job_yaml_absolute_path=None, job_api_key=None):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path, job_api_key=job_api_key)
        self.out_model_inference_url = None
        self.out_endpoint_id = None
        self.out_request_body = None
        self.out_api_key_token = self.job_api_key
        self.run_status = None

    def run(self):
        super().run()

        self.out_endpoint_id = self.launch_result.inner_id

        if self.launch_result_code != 0:
            self.output_data_dict = {
                "error": self.launch_result_code, "message": self.launch_result_message}
            print(f"{self.output_data_dict}")
            return

        endpoint_status = None
        endpoint_detail = None
        running_start_time = time.time()
        while True:
            try:
                endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(self.out_endpoint_id, self.job_api_key)
            except Exception as e:
                pass

            if endpoint_detail is not None:
                endpoint_status = endpoint_detail.status
                if endpoint_status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
                    self.run_status = JobStatus.FINISHED
                    break
                elif endpoint_status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                    self.run_status = JobStatus.FAILED
                    break
                elif endpoint_status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_KILLED:
                    self.run_status = JobStatus.FAILED
                    break
                else:
                    self.run_status = JobStatus.RUNNING

            time.sleep(10)
            if time.time() - running_start_time >= ModelDeployJob.ALLOWED_MAX_RUNNINT_TIME:
                self.run_status = ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED
                break

        if self.run_status == JobStatus.FINISHED:
            self.out_model_inference_url = endpoint_detail.inference_url
            self.out_request_body = endpoint_detail.input_json
        else:
            self.out_model_inference_url = ""
            self.out_request_body = ""

        self.output_data_dict = {"endpoint_id": self.out_endpoint_id,
                                 "inference_url": self.out_model_inference_url,
                                 "request_body": self.out_request_body,
                                 "key_token": self.out_api_key_token}

    def status(self):
        current_status = super().status()
        if current_status == JobStatus.PROVISIONING or current_status == JobStatus.FAILED:
            return current_status

        return self.run_status

    def kill(self):
        super().kill()
        FedMLModelCards.get_instance().delete_endpoint_api(self.job_api_key, self.out_endpoint_id)

