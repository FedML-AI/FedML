from fedml.workflow.jobs import Job, JobStatus
from fedml.workflow.workflow import Workflow


class HelloWorldJob(Job):
    def run(self):
        print("Hello, World!")

    def status(self):
        return JobStatus.SUCCESS

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
