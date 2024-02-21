import os
import argparse
from fedml.workflow.customized_jobs.model_deploy_job import ModelDeployJob
from fedml.workflow.workflow import Workflow


def create_deploy_workflow(job_api_key=None):
    # Define the job yaml
    working_directory = os.path.dirname(os.path.abspath(__file__))
    deploy_image_job_yaml = os.path.join(working_directory, "text2image.yaml")

    # Generate the job object
    endpoint_id = 100  # Here you need to set your own endpoint id
    deploy_image_job = ModelDeployJob(
        name="text2image",
        endpoint_id=endpoint_id,
        job_yaml_absolute_path=deploy_image_job_yaml,
        job_api_key=job_api_key,
    )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--api_key", "-k", type=str, help="API Key from the Nexus AI Platform"
    )

    args = parser.parse_args()
    workflow_status, outputs = create_deploy_workflow(job_api_key=args.api_key)
    deployed_endpoint_id = outputs.get("endpoint_id", None)
