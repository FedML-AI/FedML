

"""
Usages:
    api_key = "111sss"
    job_yaml_file = "/home/fedml/train.yaml"
    login_ret = fedml_login(api_key)
    if login_ret == 0:
        job_id, error_code, error_msg = launch_job(job_yaml_file)
        if error_code != 0:
            page_num = 1
            page_size = 100
            job_status, total_log_nums, total_log_pages, log_list = launch_log(job_id, page_num, page_size)
            print(f"job status {job_status}, total log nums {total_log_nums}, "
                  f"total log pages {total_log_pages}, log list {log_list}")
"""


def fedml_login(api_key, version="release"):
    """
    init the launch environment
    :param api_key: API Key from MLOPs
    :param version: dev, test, release
    :return int: error code (0 means successful), str: error message
    """
    return 0, ""


# inputs: yaml file
# return: job_id, error_code (0 means successful), error_message,
def launch_job(yaml_file):
    """
    launch a job
    :param yaml_file: full path of your job yaml file
    :returns: str: job id, int: error code (0 means successful), str: error message
    """
    return "0", 0, ""


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
    return "FINISHED", 100, 10, []


