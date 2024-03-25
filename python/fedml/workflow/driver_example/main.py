import os
import argparse
import traceback

import fedml
import logging

from fedml.workflow.jobs import Job, JobStatus
from fedml.workflow.workflow import Workflow


def get_full_path(file_name):
    working_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(working_directory, file_name)


class LaunchJob(Job):
    def __init__(self, name, yaml_path, api_key, version):
        super().__init__(name)
        fedml.set_env_version(version)
        self.name = name
        self.yaml_path = yaml_path
        self.api_key = api_key
        self.run_id = None
        self.launch_result = None
        self.launch_result_code = 0
        self.launch_result_message = None

    def run(self):
        try:
            self.launch_result = fedml.api.launch_job(yaml_file=self.yaml_path, api_key=self.api_key)
            if self.launch_result.run_id and int(self.launch_result.run_id) > 0:
                self.run_id = self.launch_result.run_id
            self.launch_result_code = self.launch_result.result_code
            self.launch_result_message = self.launch_result.result_message
        except Exception as e:
            self.launch_result_code = -1
            self.launch_result_message = f"Exception {e}; Traceback: {traceback.format_exc()}"

    def status(self):
        if self.launch_result_code != 0:
            self.run_status = JobStatus.FAILED
            return JobStatus.FAILED

        if self.run_id:
            try:
                _, run_status = fedml.api.run_status(run_id=self.run_id, api_key=self.api_key)
                # print(f"run_id: {self.run_id}, run_status: {run_status}")
                return JobStatus.get_job_status_from_run_status(run_status)
            except Exception as e:
                logging.error(f"Error while getting status of run {self.run_id}: {e}")
            return JobStatus.UNDETERMINED

    def kill(self):
        if self.run_id:
            try:
                return fedml.api.run_stop(run_id=self.run_id, api_key=self.api_key)
            except Exception as e:
                logging.error(f"Error while stopping run {self.run_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--api-key", "-k", type=str, help="Specify the API key for the Nexus AI Platform")
    parser.add_argument("--version", "-v", type=str, default="release",
                        help='Specify the version of the Nexus AI Platform. It should be test or release. Defaults to release')
    args = parser.parse_args()

    job_1 = LaunchJob(name="launch_job", yaml_path=get_full_path("hello_world_job.yaml"), api_key=args.api_key,
                          version=args.version)
    job_2 = LaunchJob(name="dependent_launch_job", yaml_path=get_full_path("hello_world_job.yaml"),
                          api_key=args.api_key, version=args.version)
    workflow = Workflow(name="launch_workflow", loop=False)
    workflow.add_job(job_1)
    workflow.add_job(job_2, dependencies=[job_1])
    workflow.run()

    print("graph", workflow.metadata.graph, type(workflow.metadata.graph))
    print("nodes", workflow.metadata.nodes, type(workflow.metadata.nodes))
    print("topological_order", workflow.metadata.topological_order, type(workflow.metadata.topological_order))
    print("output", workflow.get_workflow_output())
    print("loop", workflow.loop)
