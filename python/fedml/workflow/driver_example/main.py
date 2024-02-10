import logging

import fedml
import os
from fedml.workflow.jobs import Job, JobStatus
from fedml.workflow.workflow import Workflow

CURRENT_CONFIG_VERSION = "release"
CURRENT_ON_PREM_LOCAL_HOST = "localhost"
CURRENT_ON_PREM_LOCAL_PORT = 18080
MY_API_KEY = "1316b93c82da40ce90113a2ed12f0b14"


class HelloWorldJob(Job):
    def __init__(self, name):
        super().__init__(name)
        self.run_id = None

    def run(self):
        fedml.set_env_version(CURRENT_CONFIG_VERSION)
        fedml.set_local_on_premise_platform_host(CURRENT_ON_PREM_LOCAL_HOST)
        fedml.set_local_on_premise_platform_port(CURRENT_ON_PREM_LOCAL_PORT)

        working_directory = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.join(working_directory, "hello_world_job.yaml")
        result = fedml.api.launch_job(yaml_file=absolute_path, api_key=MY_API_KEY)
        if result.run_id and int(result.run_id) > 0:
            self.run_id = result.run_id

    def status(self):
        if self.run_id:
            try:
                _, run_status = fedml.api.run_status(run_id=self.run_id, api_key=MY_API_KEY)
                return JobStatus.get_job_status_from_run_status(run_status)
            except Exception as e:
                logging.error(f"Error while getting status of run {self.run_id}: {e}")
        return JobStatus.UNDETERMINED

    def kill(self):
        if self.run_id:
            try:
                return fedml.api.run_stop(run_id=self.run_id, api_key=MY_API_KEY)
            except Exception as e:
                logging.error(f"Error while stopping run {self.run_id}: {e}")


if __name__ == "__main__":
    job_1 = HelloWorldJob(name="hello_world")
    job_2 = HelloWorldJob(name="hello_world_dependent_on_job_1")
    workflow = Workflow(name="hello_world_workflow", loop=False)
    workflow.add_job(job_1)
    workflow.add_job(job_2, dependencies=[job_1])
    workflow.run()

    print("graph", workflow.metadata.graph)
    print("nodes", workflow.metadata.nodes)
    print("topological_order", workflow.metadata.topological_order)
    print("output", workflow.get_workflow_output())
    print("loop", workflow.loop)
