import logging
import traceback

from fedml.workflow.jobs import Job, JobStatus
from fedml.computing.scheduler.comm_utils import yaml_utils
from fedml.computing.scheduler.scheduler_entry.constants import Constants
import fedml


class CustomizedBaseJob(Job):
    CURRENT_ON_PREM_LOCAL_HOST = "localhost"
    CURRENT_ON_PREM_LOCAL_PORT = 18080

    def __init__(self, name, job_yaml_absolute_path=None, job_api_key=None, version="release"):
        super().__init__(name)
        self.launch_result = None
        self.run_id = None
        self.job_yaml_absolute_path = job_yaml_absolute_path
        self.job_api_key = job_api_key
        self.config_version = version
        self.local_on_prem_host = CustomizedBaseJob.CURRENT_ON_PREM_LOCAL_HOST
        self.local_on_prem_port = CustomizedBaseJob.CURRENT_ON_PREM_LOCAL_PORT
        self.launch_result_code = 0
        self.launch_result_message = None

    def run(self):
        fedml.set_env_version(self.config_version)
        fedml.set_local_on_premise_platform_host(self.local_on_prem_host)
        fedml.set_local_on_premise_platform_port(self.local_on_prem_port)

        try:
            self.launch_result = fedml.api.launch_job(yaml_file=self.job_yaml_absolute_path, api_key=self.job_api_key)
            if self.launch_result.run_id and int(self.launch_result.run_id) > 0:
                self.run_id = self.launch_result.run_id
            self.launch_result_code = self.launch_result.result_code
            self.launch_result_message = self.launch_result.result_message
        except Exception as e:
            self.launch_result_code = -1
            self.launch_result_message = f"Exception {traceback.format_exc()}"
            raise e

    def status(self):
        if self.launch_result_code != 0:
            self.run_status = JobStatus.FAILED
            return JobStatus.FAILED

        if self.run_id:
            try:
                _, run_status = fedml.api.run_status(run_id=self.run_id, api_key=self.job_api_key)
                # print(f"run_id: {self.run_id}, run_status: {run_status}")
                return JobStatus.get_job_status_from_run_status(run_status)
            except Exception as e:
                logging.error(f"Error while getting status of run {self.run_id}: {e}")
            return JobStatus.UNDETERMINED

    def kill(self):
        if self.run_id:
            try:
                return fedml.api.run_stop(run_id=self.run_id, api_key=self.job_api_key)
            except Exception as e:
                logging.error(f"Error while stopping run {self.run_id}: {e}")

    @staticmethod
    def load_yaml_config(yaml_path):
        return yaml_utils.load_yaml_config(yaml_path)

    @staticmethod
    def generate_yaml_doc(run_config_object, yaml_file):
        Constants.generate_yaml_doc(run_config_object, yaml_file)
