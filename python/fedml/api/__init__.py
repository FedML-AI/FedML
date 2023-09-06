

"""
Usages:
    import fedml
    api_key = "111sss"
    job_yaml_file = "/home/fedml/train.yaml"
    login_ret = fedml.api.fedml_login(api_key)
    if login_ret == 0:
        resource_id, error_code, error_msg = fedml.api.match_resources(job_yaml_file)
        if error_code == 0:
            job_id, error_code, error_msg = fedml.api.launch_job(job_yaml_file, resource_id=resource_id)
            if error_code == 0:
                page_num = 1
                page_size = 100
                job_status, total_log_nums, total_log_pages, log_list = fedml.api.launch_log(job_id, page_num, page_size)
                print(f"job status {job_status}, total log nums {total_log_nums}, "
                      f"total log pages {total_log_pages}, log list {log_list}")
"""
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager


def fedml_login(api_key=None, version="release"):
    """
    init the launch environment
    :param api_key: API Key from MLOPs
    :param version: dev, test, release
    :return int: error code (0 means successful), str: error message
    """
    return FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)


# inputs: yaml file
# return: resource_id, error_code (0 means successful), error_message,
def match_resources(yaml_file):
    """
    launch a job
    :param yaml_file: full path of your job yaml file
    :returns: str: resource id, int: error code (0 means successful), str: error message
    """
    return FedMLLaunchManager.get_instance().api_match_resources(yaml_file)


# inputs: yaml file, resource id
# return: job_id, error_code (0 means successful), error_message,
def launch_job(yaml_file, resource_id=None):
    """
    launch a job
    :param yaml_file: full path of your job yaml file
    :param resource_id: resource id returned from matching resources api, if you do not specify resource id,
           we will match resources based on your job yaml, and then automatically launch the job using matched resources
    :returns: str: job id, int: error code (0 means successful), str: error message
    """
    return FedMLLaunchManager.get_instance().api_launch_job(yaml_file, resource_id=resource_id, prompt=False)


# input: job id, page num, page size
# return job status, total_log_nums, total_log_pages, log list
def launch_log(job_id, page_num, page_size):
    """
    fetch logs
    :param job_id: launched job id
    :param page_num: request page num for logs
    :param page_size: request page size for logs
    :returns: str: job status, int: total log num, int: total log pages, list: log list
    """
    return FedMLLaunchManager.get_instance().api_launch_log(job_id, page_num, page_size)


