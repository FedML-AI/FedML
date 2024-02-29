import os

import fedml
from fedml.workflow.workflow import JobStatus, Workflow
from fedml.workflow.customized_jobs.model_deploy_job import ModelDeployJob
from fedml.workflow.customized_jobs.model_inference_job import ModelInferenceJob
from fedml.workflow.customized_jobs.train_job import TrainJob
from typing import List
import argparse

MY_API_KEY = ""  # Here you need to set your API key from nexus.fedml.ai


class DeployImageJob(ModelDeployJob):
    def __init__(self, name, endpoint_id=None, job_yaml_absolute_path=None, job_api_key=None):
        super().__init__(name, endpoint_id=endpoint_id, job_yaml_absolute_path=job_yaml_absolute_path,
                         job_api_key=job_api_key)

    def run(self):
        super().run()

    def status(self):
        current_status = super().status()
        if current_status == JobStatus.FINISHED:
            pass
        elif current_status == JobStatus.FAILED:
            pass
        elif current_status == JobStatus.RUNNING:
            pass
        elif current_status == JobStatus.PROVISIONING:
            pass

        return current_status

    def kill(self):
        super().kill()


class InferenceImageJob(ModelInferenceJob):
    def __init__(self, name, endpoint_id=None, job_api_key=None):
        super().__init__(name, endpoint_id=endpoint_id, job_api_key=job_api_key)
        self.run_id = None

    def run(self):
        super().run()

    def status(self):
        current_status = super().status()
        if current_status == JobStatus.FINISHED:
            pass
        elif current_status == JobStatus.FAILED:
            pass
        elif current_status == JobStatus.RUNNING:
            pass
        elif current_status == JobStatus.PROVISIONING:
            pass

        return current_status

    def kill(self):
        super().kill()


class TrainJob(TrainJob):
    def __init__(self, name, job_yaml_absolute_path=None, job_api_key=None):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path,
                         job_api_key=job_api_key)
        self.run_id = None

    def run(self):
        super().run()

    def status(self):
        current_status = super().status()
        if current_status == JobStatus.FINISHED:
            pass
        elif current_status == JobStatus.FAILED:
            pass
        elif current_status == JobStatus.RUNNING:
            pass
        elif current_status == JobStatus.PROVISIONING:
            pass

        return current_status

    def kill(self):
        super().kill()


def show_workflow_metadata(workflow):
    # After the workflow finished, print the graph, nodes and topological order
    print("graph", workflow.metadata.graph)
    print("nodes", workflow.metadata.nodes)
    print("topological_order", workflow.metadata.topological_order)
    print("loop", workflow.loop)


def create_deploy_workflow(job_api_key=None):
    # Define the job yaml
    working_directory = os.path.dirname(os.path.abspath(__file__))
    deploy_image_job_yaml = os.path.join(working_directory, "deploy_image_job.yaml")
    train_job_yaml = os.path.join(working_directory, "train_job.yaml")

    # Load the job yaml and change some config items.
    deploy_image_job_yaml_obj = DeployImageJob.load_yaml_config(deploy_image_job_yaml)
    train_job_yaml_obj = DeployImageJob.load_yaml_config(train_job_yaml)
    # deploy_image_job_yaml_obj["computing"]["resource_type"] = "A100-80GB-SXM"
    # deploy_image_job_yaml_obj["computing"]["device_type"] = "GPU"
    # DeployImageJob.generate_yaml_doc(deploy_image_job_yaml_obj, deploy_image_job_yaml)

    # Generate the job object
    endpoint_id = 100  # Here you need to set your own endpoint id
    deploy_image_job = DeployImageJob(
        name="deploy_image_job", endpoint_id=endpoint_id,
        job_yaml_absolute_path=deploy_image_job_yaml, job_api_key=job_api_key)

    # Define the workflow
    workflow = Workflow(name="deploy_workflow", loop=False)

    # Add the job object to workflow and set the dependencies (DAG based).
    workflow.add_job(deploy_image_job)

    # Run workflow
    workflow.run()

    # Get the status and result of workflow
    workflow_status = workflow.get_workflow_status()
    workflow_output = workflow.get_workflow_output()
    all_jobs_outputs = workflow.get_all_jobs_outputs()
    print(f"Final status of the workflow is as follows. {workflow_status}")
    print(f"Output of the workflow is as follows. {workflow_output}")
    print(f"Output of all jobs is as follows. {all_jobs_outputs}")

    return workflow_status, workflow_output


def create_inference_train_workflow(
        job_api_key=None, endpoint_id_list: List[int] = None, input_json=None):
    # Define the job yaml
    working_directory = os.path.dirname(os.path.abspath(__file__))
    deploy_image_job_yaml = os.path.join(working_directory, "deploy_image_job.yaml")
    train_job_yaml = os.path.join(working_directory, "train_job.yaml")

    # Load the job yaml and change some config items.
    deploy_image_job_yaml_obj = DeployImageJob.load_yaml_config(deploy_image_job_yaml)
    train_job_yaml_obj = DeployImageJob.load_yaml_config(train_job_yaml)
    # deploy_image_job_yaml_obj["computing"]["resource_type"] = "A100-80GB-SXM"
    # deploy_image_job_yaml_obj["computing"]["device_type"] = "GPU"
    # DeployImageJob.generate_yaml_doc(deploy_image_job_yaml_obj, deploy_image_job_yaml)

    # Generate the job object
    inference_jobs = list()
    for index, endpoint_id in enumerate(endpoint_id_list):
        inference_job = InferenceImageJob(
            name=f"inference_job_{index}", endpoint_id=endpoint_id, job_api_key=job_api_key)
        inference_jobs.append(inference_job)
    train_job = TrainJob(name="train_job", job_yaml_absolute_path=train_job_yaml, job_api_key=job_api_key)

    # Define the workflow
    workflow = Workflow(name="inference_workflow", loop=False)

    # Add the job object to workflow and set the dependency (DAG based).
    for index, inference_job in enumerate(inference_jobs):
        if index == 0:
            workflow.add_job(inference_job)
        else:
            workflow.add_job(inference_job, dependencies=[inference_jobs[index - 1]])
    # workflow.add_job(train_job, dependencies=[inference_jobs[-1]])

    # Set the input to the workflow
    input_json = {"text": "What is a good cure for hiccups?"} if input_json is None else input_json
    workflow.set_workflow_input(input_json)

    # Run workflow
    workflow.run()

    # Get the status and result of workflow
    workflow_status = workflow.get_workflow_status()
    workflow_output = workflow.get_workflow_output()
    all_jobs_outputs = workflow.get_all_jobs_outputs()
    print(f"Final status of the workflow is as follows. {workflow_status}")
    print(f"Output of the workflow is as follows. {workflow_output}")
    print(f"Output of all jobs is as follows. {all_jobs_outputs}")

    return workflow_status, workflow_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--deploy", "-d", nargs="*", help="Create a deploy workflow")
    parser.add_argument("--inference", "-i", nargs="*", help='Create a inference workflow')
    parser.add_argument("--endpoint_id", "-e", type=str, default=None, help='Endpoint id for inference')
    parser.add_argument("--api_key", "-k", type=str, default=MY_API_KEY, help='API Key from the Nexus AI Platform')
    parser.add_argument("--infer_json", "-ij", type=str, default=None, help='Input json data for inference')

    args = parser.parse_args()
    is_deploy = args.deploy
    if args.deploy is None:
        is_deploy = False
    else:
        is_deploy = True
    is_inference = args.inference
    if args.inference is None:
        is_inference = False
    else:
        is_inference = True

    workflow_status, outputs = None, None
    deployed_endpoint_id = args.endpoint_id
    if is_deploy:
        workflow_status, outputs = create_deploy_workflow(job_api_key=args.api_key)
        deployed_endpoint_id = outputs.get("endpoint_id", None)

    if is_inference and deployed_endpoint_id is not None:
        create_inference_train_workflow(
            job_api_key=args.api_key, endpoint_id_list=[deployed_endpoint_id, deployed_endpoint_id],
            input_json=args.infer_json)
        exit(0)
