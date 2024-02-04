import os

from fedml.workflow.workflow import JobStatus, Workflow
from fedml.workflow.customized_jobs.deploy_job import DeployJob
from fedml.workflow.customized_jobs.train_job import TrainJob

CURRENT_CONFIG_VERSION = "release"
CURRENT_ON_PREM_LOCAL_HOST = "localhost"
CURRENT_ON_PREM_LOCAL_PORT = 18080
MY_API_KEY = "" # Here you need to set your API key from nexus.fedml.ai


class DeployImageJob(DeployJob):
    def __init__(self, name, job_yaml_absolute_path=None):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path,
                         job_api_key=MY_API_KEY, config_version=CURRENT_CONFIG_VERSION,
                         local_on_prem_host=CURRENT_ON_PREM_LOCAL_HOST,
                         local_on_prem_port=CURRENT_ON_PREM_LOCAL_PORT)

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


class Deploy3DJob(DeployJob):
    def __init__(self, name, job_yaml_absolute_path=None):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path,
                         job_api_key=MY_API_KEY, config_version=CURRENT_CONFIG_VERSION,
                         local_on_prem_host=CURRENT_ON_PREM_LOCAL_HOST,
                         local_on_prem_port=CURRENT_ON_PREM_LOCAL_PORT)
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
    def __init__(self, name, job_yaml_absolute_path=None):
        super().__init__(name, job_yaml_absolute_path=job_yaml_absolute_path,
                         job_api_key=MY_API_KEY, config_version=CURRENT_CONFIG_VERSION,
                         local_on_prem_host=CURRENT_ON_PREM_LOCAL_HOST,
                         local_on_prem_port=CURRENT_ON_PREM_LOCAL_PORT)
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


if __name__ == "__main__":
    # Define the job yaml
    working_directory = os.path.dirname(os.path.abspath(__file__))
    deploy_image_job_yaml = os.path.join(working_directory, "deploy_image_job.yaml")
    deploy_3d_job_yaml = os.path.join(working_directory, "deploy_3d_job.yaml")
    train_job_yaml = os.path.join(working_directory, "train_job.yaml")

    # Load the job yaml and change some config items.
    deploy_image_job_yaml_obj = DeployImageJob.load_yaml_config(deploy_image_job_yaml)
    deploy_3d_job_yaml_obj = DeployImageJob.load_yaml_config(deploy_3d_job_yaml)
    train_job_yaml_obj = DeployImageJob.load_yaml_config(train_job_yaml)
    # deploy_image_job_yaml_obj["computing"]["resource_type"] = "A100-80GB-SXM"
    # deploy_image_job_yaml_obj["computing"]["device_type"] = "GPU"
    # DeployImageJob.generate_yaml_doc(deploy_image_job_yaml_obj, deploy_image_job_yaml)

    # Generate the job object
    deploy_image_job = DeployImageJob(name="deploy_image_job", job_yaml_absolute_path=deploy_image_job_yaml)
    deploy_3d_job = Deploy3DJob(name="deploy_3d_job", job_yaml_absolute_path=deploy_3d_job_yaml)
    train_job = TrainJob(name="train_job", job_yaml_absolute_path=train_job_yaml)

    # Define the workflow
    workflow = Workflow(name="workflow_with_multi_jobs", loop=False)

    # Add the job object to workflow and set the dependency (DAG based).
    workflow.add_job(deploy_image_job)
    #workflow.add_job(deploy_3d_job, dependencies=[deploy_image_job])
    workflow.add_job(train_job, dependencies=[deploy_image_job])

    # Run workflow
    workflow.run()

    # After the workflow finished, print the graph, nodes and topological order
    print("graph", workflow.metadata.graph)
    print("nodes", workflow.metadata.nodes)
    print("topological_order", workflow.metadata.topological_order)
    print("loop", workflow.loop)
