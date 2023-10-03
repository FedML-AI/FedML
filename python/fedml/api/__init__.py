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
from fedml.api.modules import launch, utils, job, build, device, logs, diagnosis, model, cluster, run
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterModelList


def fedml_login(api_key=None):
    """
    init the launch environment
    :param api_key: API Key from MLOPs
    :return int: error code (0 means successful), str: error message
    """
    return utils.login(api_key)


# inputs: yaml file
# return: resource_id, error_code (0 means successful), error_message,
def match_resources(yaml_file, cluster=""):
    """
    launch a job
    :param yaml_file: full path of your job yaml file
    :returns: str: resource id, int: error code (0 means successful), str: error message
    """
    return utils.match_resources(yaml_file, cluster, prompt=False)


# inputs: yaml file, resource id
# return: job_id, error_code (0 means successful), error_message,
def launch_job(yaml_file, cluster="", api_key=None, resource_id=None, prompt=True):
    """
    launch a job
    :param yaml_file: full path of your job yaml file
    :param resource_id: resource id returned from matching resources api, if you do not specify resource id,
           we will match resources based on your job yaml, and then automatically launch the job using matched resources
    :returns: str: job id, int: error code (0 means successful), str: error message
    """
    return launch.job(yaml_file, api_key, resource_id, cluster, prompt=prompt)


def job_stop(job_id, platform="falcon", api_key=None):
    return job.stop(job_id, platform, api_key)


def job_list(job_name, job_id=None, platform="falcon", api_key=None):
    return job.list_job(job_name, job_id, platform, api_key)


def job_status(job_name, job_id, platform, api_key):
    return job.status(job_name, job_id, platform, api_key)


def job_logs(job_id, page_num, page_size, need_all_logs=False, platform="falcon", api_key=None):
    """
    fetch logs

    :param str job_id: launched job id
    :param int page_num: request page num for logs
    :param int page_size: request page size for logs
    :param bool need_all_logs: boolean value representing if all logs are needed. Default is False
    :param str platform: The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon,
                         launch). Default is falcon
    :param str api_key: API Key from MLOPs. Not needed if already configured once

    :returns: str: job_status, int: total_log_lines, int: total_log_pages, List[str]: log_list, FedMLJobLogModelList:
    logs

    :rtype: Tuple[str, int, int, List[str], FedMLJobLogModelList]
    """
    return job.logs(job_id, page_num, page_size, need_all_logs, platform, api_key)


def cluster_list(cluster_names=(), api_key=None) -> FedMLClusterModelList:
    return cluster.list_clusters(cluster_names=cluster_names, api_key=api_key)

def cluster_exists(cluster_name:str, api_key:str=None) -> bool:
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


def fedml_build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore):
    return build.build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore)


def login(userid, client, server,
          api_key, role, runner_cmd, device_id, os_name,
          docker, docker_rank):
    device_bind(userid, client, server,
                api_key, role, runner_cmd, device_id, os_name,
                docker, docker_rank)


def logout(client, server, docker, docker_rank):
    device_unbind(client, server, docker, docker_rank)


def device_bind(userid, client, server,
                api_key, role, runner_cmd, device_id, os_name,
                docker, docker_rank):
    device.bind(userid, client, server,
                api_key, role, runner_cmd, device_id, os_name,
                docker, docker_rank)


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


def model_push(name, model_storage_url, model_net_url, user, api_key):
    model.push(name, model_storage_url, model_net_url, user, api_key)


def model_pull(name, user, api_key):
    model.pull(name, user, api_key)


def model_deploy(local, name, master_ids, worker_ids, user_id, api_key, config_file):
    model.deploy(local, name, master_ids, worker_ids, user_id, api_key, config_file)


def model_info(name):
    model.info(name)


def model_run(name, data):
    model.run(name, data)


def run_command(cmd, cluster_name, api_key=None):
    return run.command(cmd, cluster_name, api_key)
