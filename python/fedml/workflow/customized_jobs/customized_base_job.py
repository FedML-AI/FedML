import logging

from fedml.workflow.jobs import Job, JobStatus
from fedml.computing.scheduler.comm_utils import yaml_utils
from fedml.computing.scheduler.scheduler_entry.constants import Constants
import fedml


class CustomizedBaseJob(Job):
    def __init__(self, name, job_yaml_absolute_path=None, job_api_key=None, config_version=None,
                 local_on_prem_host="localhost", local_on_prem_port=80):
        super().__init__(name)
        self.run_id = None
        self.job_yaml_absolute_path = job_yaml_absolute_path
        self.job_api_key = job_api_key
        self.config_version = config_version
        self.local_on_prem_host = local_on_prem_host
        self.local_on_prem_port = local_on_prem_port

    def append_input(self, job_name):
        pass

    def get_input(self, job_name):
        pass

    def generate_output(self):
        pass

    def run(self):
        fedml.set_env_version(self.config_version)
        fedml.set_local_on_premise_platform_host(self.local_on_prem_host)
        fedml.set_local_on_premise_platform_port(self.local_on_prem_port)

        result = fedml.api.launch_job(yaml_file=self.job_yaml_absolute_path, api_key=self.job_api_key)
        if result.run_id and int(result.run_id) > 0:
            self.run_id = result.run_id

    def status(self):
        if self.run_id:
            try:
                _, run_status = fedml.api.run_status(run_id=self.run_id, api_key=self.job_api_key)
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
