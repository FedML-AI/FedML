"""
Usages:
    import fedml
    api_key = "111sss"
    job_yaml_file = "/home/fedml/train.yaml"
    login_ret = fedml.api.fedml_login(api_key)
    if login_ret == 0:
        launch_result = fedml.api.launch_job(job_yaml_file)
        if launch_result.result_code == 0:
            page_num = 1
            page_size = 100
            log_result = fedml.api.run_logs(launch_result.run_id, page_num, page_size)
            print(f"run status {run_log_result.run_status}, total log nums {log_result.total_log_lines}, "
                  f"total log pages {log_result.total_log_pages}, log list {log_result.log_line_list}")
"""
from typing import List, Tuple

from fedml.api.modules import launch, utils, build, device, logs, diagnosis, cluster, run, train, federate, \
    model as model_module   # Since "model" has conflict with one of the input parameters, we need to rename it
from fedml.api.modules.launch import FeatureEntryPoint
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterModelList
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunStartedModel, FedMLGpuDevices, \
    FedMLRunModelList, FeatureEntryPoint


def fedml_login(api_key: str = None):
    """
    Login to FedML AI Nexus Platform

    Args:
        api_key:  API key from FedML AI Nexus Platform (Default value = None)

    Returns:
        A tuple of error_code and error_msg.
        error_code is 0 if login is successful, else -1
    """
    return utils.fedml_login(api_key)


def launch_job(
        yaml_file: str, api_key: str = None, resource_id: str = None, device_server: str = None,
        device_edges: List[str] = None,
        feature_entry_point: FeatureEntryPoint = FeatureEntryPoint.FEATURE_ENTRYPOINT_API) -> launch.LaunchResult:
    """
    Launch a job on the FedML AI Nexus platform

    Args:
        yaml_file: Full path of your job yaml file.
        api_key: Your API key (if not configured already). (Default value = None)
        resource_id:
            Specific `resource_id` to use. Typically, you won't need to specify a specific `resource_id`.
            Instead, we will match resources based on your job yaml, and then automatically launch the job
            using matched resources.
        device_server:
            `device_server` to use. Only needed when you want to launch a federated learning job with specific
            `device_server` and `device_edges`
        device_edges:
            List of `device_edges` to use. Only needed when you want to launch a federated learning job
            with specific `device_server` and `device_edges`
        feature_entry_point:
            Entry point where you launch a job. Default entry point is from API.

    Returns:
        LaunchResult object with the following attributes

            result_code:
                API result code. `0` means success.
            result_msg:
                API status message.
            run_id:
                Run ID of the launched job.
            project_id:
                Project Id of the launched job. This is default assigned if not specified in your job yaml file
            inner_id:
                Serving endpoint id of launched job. Only applicable for Deploy / Serve Job tasks,
                and will be `None` otherwise.
    """

    return launch.job(yaml_file, api_key, resource_id, device_server, device_edges,
                      feature_entry_point=feature_entry_point)


def launch_job_on_cluster(
        yaml_file: str, cluster: str, api_key: str = None, resource_id: str = None,
        device_server: str = None, device_edges: List[str] = None,
        feature_entry_point: FeatureEntryPoint = FeatureEntryPoint.FEATURE_ENTRYPOINT_API) -> launch.LaunchResult:
    """
    Launch a job on a cluster on the FedML AI Nexus platform

    Args:
        yaml_file: Full path of your job yaml file.
        cluster: Cluster name to use. If a cluster with provided name doesn't exist, one will be created.
        api_key: Your API key (if not configured already).
        resource_id: Specific `resource_id` to use. Typically, you won't need to specify a specific `resource_id`. Instead, we will match resources based on your job yaml, and then automatically launch the job using matched resources.
        device_server: `device_server` to use. Only needed when you want to launch a federated learning job with specific `device_server` and `device_edges`
        device_edges: List of `device_edges` to use. Only needed when you want to launch a federated learning job with specific `device_server` and `device_edges`
        feature_entry_point: Entry point where you launch a job. Default entry point is from API.
    Returns:
        LaunchResult object with the following attributes

            result_code:
                API result code. `0` means success.
            result_msg:
                API status message.
            run_id:
                Run ID of the launched job.
            project_id:
                Project Id of the launched job. This is default assigned if not specified in your job yaml file
            inner_id:
                Serving endpoint id of launched job. Only applicable for Deploy / Serve Job tasks,
                and will be `None` otherwise.
    """

    return launch.job_on_cluster(yaml_file=yaml_file, cluster=cluster, api_key=api_key, resource_id=resource_id,
                                 device_server=device_server, device_edges=device_edges,
                                 feature_entry_point=feature_entry_point)


def run_stop(run_id: str, platform: str = "falcon", api_key: str = None) -> bool:
    return run.stop(run_id=run_id, platform=platform, api_key=api_key)


def run_list(run_name: str, run_id: str = None, platform: str = "falcon", api_key: str = None) -> FedMLRunModelList:
    return run.list_run(run_name=run_name, run_id=run_id, platform=platform, api_key=api_key)


def run_status(run_name: str, run_id: str = None, platform: str = "falcon", api_key: str = None) -> (
FedMLRunModelList, str):
    return run.status(run_name=run_name, run_id=run_id, platform=platform, api_key=api_key)


def run_logs(run_id: str, page_num: int = 1, page_size: int = 10, need_all_logs: bool = False, platform: str = "falcon",
             api_key: str = None) -> run.RunLogResult:
    return run.logs(run_id=run_id, page_num=page_num, page_size=page_size, need_all_logs=need_all_logs,
                    platform=platform, api_key=api_key)


def cluster_list(cluster_names: Tuple[str] = (), api_key: str = None) -> FedMLClusterModelList:
    return cluster.list_clusters(cluster_names=cluster_names, api_key=api_key)


def cluster_exists(cluster_name: str, api_key: str = None) -> bool:
    return cluster.exists(cluster_name=cluster_name, api_key=api_key)


def cluster_status(cluster_name: str, api_key: str = None) -> (str, FedMLClusterModelList):
    return cluster.status(cluster_name=cluster_name, api_key=api_key)


def cluster_start(cluster_names: Tuple[str], api_key: str = None) -> bool:
    return cluster.start(cluster_names=cluster_names, api_key=api_key)


def cluster_startall(api_key: str = None) -> bool:
    return cluster.start(cluster_names=(), api_key=api_key)

def cluster_stop(cluster_names: Tuple[str], api_key: str = None) -> bool:
    return cluster.stop(cluster_names=cluster_names, api_key=api_key)


def cluster_stopall(api_key: str = None) -> bool:
    return cluster.stop(cluster_names=(), api_key=api_key)


def cluster_autostop(cluster_id:int, time:int, api_key: str = None) -> bool:
    return cluster.autostop(cluster_id=cluster_id, time=time, api_key=api_key)


def cluster_kill(cluster_names: Tuple[str], api_key: str = None) -> bool:
    return cluster.kill(cluster_names=cluster_names, api_key=api_key)


def cluster_killall(api_key=None) -> bool:
    return cluster.kill(cluster_names=(), api_key=api_key)


def fedml_build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore):
    return build.build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore)


def login(api_key, computing, server, supplier):
    device_bind(api_key, computing, server, supplier)


def logout(computing, server):
    device_unbind(computing, server)


def device_bind(api_key, computing, server, supplier):
    device.bind(api_key, computing, server, supplier)


def device_unbind(computing, server):
    device.unbind(computing, server)


def resource_type():
    device.resource_type()


def fedml_logs(client, server, docker, docker_rank):
    logs.log(client, server, docker, docker_rank)


def fedml_diagnosis(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
                    mqtt_s3_backend_run_id):
    diagnosis.diagnose(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
                       mqtt_s3_backend_run_id)


def model_create(name, model, model_config):
    model_module.create(name, model, model_config)


def model_delete(name, local):
    model_module.delete(name, local)


def model_list(name, local):
    model_module.list_models(name, local)


def model_package(name):
    model_module.package(name)


def model_push(name, model_storage_url):
    model_module.push(name, model_storage_url)


def model_pull(name):
    model_module.pull(name)


def model_deploy(name, local, master_ids, worker_ids, use_remote):
    model_module.deploy(name, local, master_ids, worker_ids, use_remote)


def train_build(job_yaml_file, dest_folder):
    return train.build_with_job_yaml(job_yaml_file, dest_folder=dest_folder)


def federate_build(job_yaml_file, dest_folder):
    return federate.build_with_job_yaml(job_yaml_file, dest_folder=dest_folder)
