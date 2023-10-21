"""
Usages:
    import fedml
    api_key = "111sss"
    job_yaml_file = "/home/fedml/train.yaml"
    login_ret = fedml.api.fedml_login(api_key)
    if login_ret == 0:
        resource_id, project_id, error_code, error_msg = fedml.api.match_resources(job_yaml_file)
        if error_code == 0:
            job_id, project_id, error_code, error_msg = fedml.api.launch_job(job_yaml_file, resource_id=resource_id)
            if error_code == 0:
                page_num = 1
                page_size = 100
                job_status, total_log_nums, total_log_pages, log_list = fedml.api.launch_log(job_id, page_num, page_size)
                print(f"job status {job_status}, total log nums {total_log_nums}, "
                      f"total log pages {total_log_pages}, log list {log_list}")
"""
from typing import List

from fedml.api.modules import launch, utils, build, device, logs, diagnosis, model, cluster, run, train, federate
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterModelList
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunStartedModel, FedMLGpuDevices


def fedml_login(api_key=None):
    """
    init the launch environment
    :param api_key: API Key from MLOPs
    :return int: error code (0 means successful), str: error message
    """
    return utils.fedml_login(api_key)


# inputs: yaml file, resource id
# return: job_id, error_code (0 means successful), error_message,
def launch_job(yaml_file, api_key=None, resource_id=None, device_server=None, device_edges=None):
    """
    launch a job
    :param api_key:
    :param yaml_file: full path of your job yaml file
    :param resource_id: resource id returned from matching resources api, if you do not specify resource id,
           we will match resources based on your job yaml, and then automatically launch the job using matched resources
    :returns: str: job id, int: error code (0 means successful), str: error message
    """
    return launch.job(yaml_file, api_key, resource_id, device_server, device_edges)


def launch_job_on_cluster(yaml_file, cluster, api_key=None, resource_id=None, prompt=True):
    return launch.job_on_cluster(yaml_file=yaml_file, cluster=cluster, api_key=api_key, resource_id=resource_id)


def start_created_run(create_run_result: FedMLRunStartedModel, api_key=None, device_server="", device_edges=""):
    return launch.run(create_run_result=create_run_result, api_key=api_key, device_server=device_server,
                      device_edges=device_edges)


def run_stop(run_id: str, platform: str = "falcon", api_key: str = None):
    return run.stop(run_id=run_id, platform=platform, api_key=api_key)


def run_list(run_name, run_id=None, platform="falcon", api_key=None):
    return run.list_run(run_name=run_name, run_id=run_id, platform=platform, api_key=api_key)


def run_status(run_name, run_id, platform, api_key):
    return run.status(run_name=run_name, run_id=run_id, platform=platform, api_key=api_key)


def run_logs(run_id, page_num, page_size, need_all_logs=False, platform="falcon", api_key=None):
    """
    fetch logs

    :param str run_id: launched run id
    :param int page_num: request page num for logs
    :param int page_size: request page size for logs
    :param bool need_all_logs: boolean value representing if all logs are needed. Default is False
    :param str platform: The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon,
                         launch). Default is falcon
    :param str api_key: API Key from MLOPs. Not needed if already configured once

    :returns: str: run_status, int: total_log_lines, int: total_log_pages, List[str]: log_list, FedMLRunLogModelList:
    logs

    :rtype: Tuple[str, int, int, List[str], FedMLRunLogModelList]
    """
    return run.logs(run_id=run_id, page_num=page_num, page_size=page_size, need_all_logs=need_all_logs,
                    platform=platform, api_key=api_key)


def cluster_list(cluster_names=(), api_key=None) -> FedMLClusterModelList:
    return cluster.list_clusters(cluster_names=cluster_names, api_key=api_key)


def cluster_exists(cluster_name: str, api_key: str = None) -> bool:
    return cluster.exists(cluster_name=cluster_name, api_key=api_key)


def cluster_status(cluster_name, api_key=None) -> FedMLClusterModelList:
    return cluster.status(cluster_name=cluster_name, api_key=api_key)


def cluster_start(cluster_names, api_key=None) -> bool:
    return cluster.start(cluster_names=cluster_names, api_key=api_key)


def cluster_startall(api_key=None) -> bool:
    return cluster.start(cluster_names=(), api_key=api_key)


def cluster_stop(cluster_names, api_key=None) -> bool:
    return cluster.stop(cluster_names=cluster_names, api_key=api_key)


def cluster_stopall(api_key=None) -> bool:
    return cluster.stop(cluster_names=(), api_key=api_key)


def cluster_kill(cluster_names, api_key=None) -> bool:
    return cluster.kill(cluster_names=cluster_names, api_key=api_key)


def cluster_killall(api_key=None) -> bool:
    return cluster.kill(cluster_names=(), api_key=api_key)


def confirm_cluster_and_start_run(run_id: str, cluster_id: str, gpu_matched: List[FedMLGpuDevices], api_key: str = None):
    return cluster.confirm_and_start(run_id=run_id, cluster_id=cluster_id, gpu_matched=gpu_matched, api_key=api_key)


def fedml_build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore):
    return build.build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore)


def login(userid, client, server,
          api_key, role, runner_cmd, device_id, os_name,
          docker, docker_rank, infer_host,
          redis_addr, redis_port, redis_password):
    device_bind(userid, client, server,
                api_key, role, runner_cmd, device_id, os_name,
                docker, docker_rank, infer_host,
                redis_addr, redis_port, redis_password)


def logout(client, server, docker, docker_rank):
    device_unbind(client, server, docker, docker_rank)


def device_bind(userid, client, server,
                api_key, role, runner_cmd, device_id, os_name,
                docker, docker_rank, infer_host,
                redis_addr, redis_port, redis_password):
    device.bind(userid, client, server,
                api_key, role, runner_cmd, device_id, os_name,
                docker, docker_rank, infer_host,
                redis_addr, redis_port, redis_password)


def device_unbind(client, server, docker, docker_rank):
    device.unbind(client, server, docker, docker_rank)


def resource_type():
    device.resource_type()


def fedml_logs(client, server, docker, docker_rank):
    logs.log(client, server, docker, docker_rank)


def fedml_diagnosis(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
                    mqtt_s3_backend_run_id):
    diagnosis.diagnose(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
                       mqtt_s3_backend_run_id)


def model_create(name, config_file):
    model.create(name, config_file)


def model_delete(name):
    model.delete(name)


def model_list(name):
    model.list_models(name)


def model_list_remote(name, user, api_key):
    model.list_remote(name, user, api_key)


def model_package(name):
    model.package(name)


def model_push(name, model_storage_url, model_net_url, api_key):
    model.push(name, model_storage_url, model_net_url, api_key)


def model_pull(name, user, api_key):
    model.pull(name, user, api_key)


def model_deploy(local, name, master_ids, worker_ids):
    model.deploy(local, name, master_ids, worker_ids)


def model_info(name):
    model.info(name)


def model_run(name, data):
    model.run(name, data)


def train_build(source_folder, entry_point, entry_args, config_folder, dest_folder, ignore,
                model_name, model_cache_path, input_dim, output_dim, dataset_name, dataset_type, dataset_path):
    return train.build(source_folder, entry_point, entry_args, config_folder, dest_folder, ignore,
                       model_name, model_cache_path, input_dim, output_dim, dataset_name, dataset_type, dataset_path)


def federate_build(source_folder, entry_point, config_folder, dest_folder, ignore,
                   model_name, model_cache_path, input_dim, output_dim, dataset_name, dataset_type, dataset_path):
    return federate.build(source_folder, entry_point, config_folder, dest_folder, ignore,
                          model_name, model_cache_path, input_dim, output_dim, dataset_name, dataset_type, dataset_path)
