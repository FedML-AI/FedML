import os
import uuid

from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.workflow.jobs import JobStatus
import time
from fedml.workflow.workflow_mlops_api import WorkflowMLOpsApi


class ModelDeployJob(CustomizedBaseJob):
    ALLOWED_MAX_RUNNING_TIME = 2 * 60 * 60

    def __init__(self, name, endpoint_name=None, job_yaml_absolute_path=None, job_api_key=None):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path, job_api_key=job_api_key)
        self.out_model_inference_url = None
        self.in_endpoint_id = None
        self.out_endpoint_id = None
        self.in_endpoint_name = endpoint_name
        self.out_endpoint_name = endpoint_name
        self.out_request_body = None
        self.out_api_key_token = self.job_api_key
        self.run_status = None
        self.job_yaml_absolute_path_origin = self.job_yaml_absolute_path
        self.job_yaml_dir = os.path.dirname(self.job_yaml_absolute_path)
        self.job_yaml_absolute_path_for_launch = os.path.join(
            self.job_yaml_dir, f"{str(uuid.uuid4())}.yaml")

    def run(self):
        job_yaml_obj = self.load_yaml_config(self.job_yaml_absolute_path_origin)
        job_yaml_obj["serving_args"] = dict()
        job_yaml_obj["serving_args"]["endpoint_name"] = self.in_endpoint_name
        self.generate_yaml_doc(job_yaml_obj, self.job_yaml_absolute_path_for_launch)
        self.job_yaml_absolute_path = self.job_yaml_absolute_path_for_launch

        super().run()

        os.remove(self.job_yaml_absolute_path_for_launch)

        self.run_id = self.launch_result.inner_id
        dependency_list = list()
        for dep in self.dependencies:
            dependency_list.append(dep)
        result = WorkflowMLOpsApi.add_run(
            workflow_id=self.workflow_id, job_name=self.name, run_id=self.run_id,
            dependencies=dependency_list, api_key=self.job_api_key
        )
        if not result:
            raise Exception("Unable to upload job metadata to the backend.")

        if self.launch_result_code != 0:
            self.output_data_dict = {
                "error": self.launch_result_code, "message": self.launch_result_message}
            print(f"{self.output_data_dict}")
            return

        self.out_endpoint_id = self.launch_result.inner_id

        endpoint_status = None
        endpoint_detail = None
        running_start_time = time.time()
        while True:
            try:
                endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(
                    endpoint_id=self.out_endpoint_id, user_api_key=self.job_api_key)
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
            if time.time() - running_start_time >= ModelDeployJob.ALLOWED_MAX_RUNNING_TIME:
                self.run_status = ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED
                break

        if self.run_status == JobStatus.FINISHED:
            self.out_model_inference_url = endpoint_detail.inference_url
            self.out_request_body = endpoint_detail.input_json
            self.out_endpoint_name = endpoint_detail.endpoint_name
        else:
            self.out_model_inference_url = ""
            self.out_request_body = ""

        self.output_data_dict = {"endpoint_id": self.out_endpoint_id,
                                 "endpoint_name": self.out_endpoint_name,
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
