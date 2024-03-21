import argparse

from fedml.workflow.customized_jobs.customized_base_job import CustomizedBaseJob
from fedml.workflow.workflow import Workflow
import os


class HelloWorldJob(CustomizedBaseJob):
    def __init__(self, name, yaml_path, api_key, version):
        super().__init__(name=name, job_yaml_absolute_path=yaml_path, job_api_key=api_key, version=version)


def get_full_path(file_name):
    working_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(working_directory, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--api-key", "-k", type=str, help="Specify the API key for the Nexus AI Platform")
    parser.add_argument("--version", "-v", type=str, default="release",
                        help='Specify the version of the Nexus AI Platform. It should be test or release. Defaults to release')
    args = parser.parse_args()

    job_1 = HelloWorldJob(name="hello_world", yaml_path=get_full_path("hello_world_job.yaml"), api_key=args.api_key,
                          version=args.version)
    job_2 = HelloWorldJob(name="hello_world_dependent_on_job_1", yaml_path=get_full_path("hello_world_job.yaml"),
                          api_key=args.api_key, version=args.version)
    workflow = Workflow(name="hello_world_workflow", loop=False)
    workflow.add_job(job_1)
    workflow.add_job(job_2, dependencies=[job_1])
    workflow.run()

    print("graph", workflow.metadata.graph, type(workflow.metadata.graph))
    print("nodes", workflow.metadata.nodes, type(workflow.metadata.nodes))
    print("topological_order", workflow.metadata.topological_order, type(workflow.metadata.topological_order))
    print("output", workflow.get_workflow_output())
    print("loop", workflow.loop)
