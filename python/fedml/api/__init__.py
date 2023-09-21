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
from fedml.api.modules import launch, utils, job, build, device, logs, diagnosis, model


def fedml_login(api_key=None, version="release"):
    """
    init the launch environment
    :param api_key: API Key from MLOPs
    :param version: dev, test, release
    :return int: error code (0 means successful), str: error message
    """
    return utils.login(api_key, version)


# inputs: yaml file
# return: resource_id, error_code (0 means successful), error_message,
def match_resources(yaml_file):
    """
    launch a job
    :param yaml_file: full path of your job yaml file
    :returns: str: resource id, int: error code (0 means successful), str: error message
    """
    return utils.match_resources(yaml_file, prompt=False)


# inputs: yaml file, resource id
# return: job_id, error_code (0 means successful), error_message,
def launch_job(yaml_file, version="release", api_key=None, resource_id=None, prompt=True):
    """
    launch a job
    :param yaml_file: full path of your job yaml file
    :param resource_id: resource id returned from matching resources api, if you do not specify resource id,
           we will match resources based on your job yaml, and then automatically launch the job using matched resources
    :param version: version of MLOps platform. It should be dev, test or release
    :returns: str: job id, int: error code (0 means successful), str: error message
    """
    return launch.job(yaml_file, api_key, version, resource_id, prompt=prompt)


# input: job id, page num, page size, need_all_logs
# return job status, total_log_nums, total_log_pages, log list
def launch_log(job_id, page_num, page_size, version="release", api_key=None, need_all_logs=False):
    """
    fetch logs
    :param job_id: launched job id
    :param page_num: request page num for logs
    :param page_size: request page size for logs
    :param need_all_logs: boolean value representing if all logs are needed
    :returns: str: job status, int: total log num, int: total log pages, list: log list
    """
    return launch.log(job_id, version, api_key, page_num, page_size, need_all_logs)


def stop_job(job_id, version, platform="falcon", api_key=None, show_hint_texts=True):
    return job.stop(job_id, version, platform, api_key, show_hint_texts)


def list_jobs(version, job_name, job_id=None, platform="falcon", api_key=None):
    return job.lists(version, job_name, job_id, platform, api_key)


def fedml_build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore):
    return build.build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore)


def login(userid, version, client, server,
          api_key, local_server, role, runner_cmd, device_id, os_name,
          docker, docker_rank):
    device_bind(userid, version, client, server,
                api_key, local_server, role, runner_cmd, device_id, os_name,
                docker, docker_rank)


def logout(client, server, docker, docker_rank):
    device_unbind(client, server, docker, docker_rank)


def device_bind(userid, version, client, server,
                api_key, local_server, role, runner_cmd, device_id, os_name,
                docker, docker_rank):
    device.bind(userid, version, client, server,
                api_key, local_server, role, runner_cmd, device_id, os_name,
                docker, docker_rank)


def device_unbind(client, server, docker, docker_rank):
    device.unbind(client, server, docker, docker_rank)


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


def model_add_files(name, path):
    model.add_files(name, path)


def model_remove_files(name, file):
    model.remove_files(name, file)


def model_list(name):
    model.list_models(name)


def model_list_remote(name, user, api_key, version, local_server):
    model.list_remote(name, user, api_key, version, local_server)


def model_package(name):
    model.package(name)


def model_push(name, model_storage_url, model_net_url, user, api_key, version, local_server):
    model.push(name, model_storage_url, model_net_url, user, api_key, version, local_server)


def model_pull(name, user, api_key, version, local_server):
    model.pull(name, user, api_key, version, local_server)


def model_deploy(local, name, master_ids, worker_ids, user_id, api_key):
    model.deploy(local, name, master_ids, worker_ids, user_id, api_key)


def model_info(name):
    model.info(name)


def model_run(name, data):
    model.run(name, data)


def resource_type(version):
    model.resource_type(version)