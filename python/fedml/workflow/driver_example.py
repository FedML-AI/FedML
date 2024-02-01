import fedml
import os
from fedml.workflow.jobs import Job, JobStatus
from fedml.workflow.workflow import Workflow


class HelloWorldJob(Job):
    def __init__(self, name):
        super().__init__(name)
        self.run_id = None
        self.status = JobStatus.UNDETERMINED

    def run(self):
        fedml.set_env_version("test")
        current_working_directory = os.getcwd()
        absolute_path = os.path.join(current_working_directory, "hello_world_job.yaml")
        result = fedml.api.launch_job(yaml_file=absolute_path, api_key="30d1bbcae9ec48ffa314caa8e944d187")
        if result.run_id and int(result.run_id) > 0:
            self.run_id = result.run_id
        else:
            self.status = JobStatus.FAILED


    def status(self):
        if self.run_id:
            try:
                run_status = fedml.api.run_status(run_id=self.run_id, api_key="30d1bbcae9ec48ffa314caa8e944d187")
                if run_status:
                    self.status = JobStatus.SUCCESS
                else:
                    self.status = JobStatus.FAILED

    def kill(self):
        pass


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
    print("loop", workflow.loop)
