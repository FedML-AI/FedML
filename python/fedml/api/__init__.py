"""
Usages:
    import fedml
    api_key = "111sss"
    job_yaml_file = "/home/fedml/train.yaml"
    login_ret = fedml.api.fedml_login(api_key)
    if login_ret == 0:
        launch_result = fedml.api.job(job_yaml_file)
        if launch_result.result_code == 0:
            page_num = 1
            page_size = 100
            log_result = fedml.api.run_logs(launch_result.run_id, page_num, page_size)
            print(f"run status {run_log_result.run_status}, total log nums {log_result.total_log_lines}, "
                  f"total log pages {log_result.total_log_pages}, log list {log_result.log_line_list}")
"""
from typing import Tuple

from fedml.api.modules import utils, build, device, logs, diagnosis, cluster, run, train, federate, \
    model as model_module   # Since "model" has conflict with one of the input parameters, we need to rename it
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterModelList
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunModelList


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
