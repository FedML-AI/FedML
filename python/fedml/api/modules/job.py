from typing import List

from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobLogModelList
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager, FedMLJobManager
from fedml.computing.scheduler.comm_utils.security_utils import get_api_key


def stop(job_id, platform, api_key):
    authenticate(api_key)

    if not platform_is_valid(platform):
        return

    is_stopped = FedMLJobManager.get_instance().stop_job(platform, api_key, job_id)
    return is_stopped


def list_job(job_name, job_id, platform, api_key):
    if not _authenticated_and_validated_platform(api_key, platform):
        return

    job_list_obj = FedMLJobManager.get_instance().list_job(platform=platform, project_name=None, job_name=job_name,
                                                           user_api_key=get_api_key(), job_id=job_id)

    return job_list_obj


def status(job_name, job_id, platform, api_key):
    if not _authenticated_and_validated_platform(api_key, platform):
        return

    if job_name is None and job_id is None:
        raise Exception("Please specify either job name or job id.")

    job_status = None
    job_list_obj = FedMLJobManager.get_instance().list_job(platform=platform, project_name=None, job_name=job_name,
                                                           user_api_key=get_api_key(), job_id=job_id)

    if job_list_obj is not None:
        if len(job_list_obj.job_list) > 1:
            raise Exception("Found more than one jobs for the specified job name or job id.")

        job_status = job_list_obj.job_list[0].status

    return job_list_obj, job_status


# input: job_id, page_num, page_size, need_all_logs, platform, api_key
# return job status, total_log_lines, total_log_pages, log_list, logs
def logs(job_id, page_num, page_size, need_all_logs, platform, api_key) -> (
str, int, int, List[str], FedMLJobLogModelList):
    if job_id is None:
        raise Exception("Please specify job id.")

    _, job_status = status(job_name=None, job_id=job_id, platform=platform, api_key=api_key)

    total_log_nums, total_log_pages, log_line_list, job_logs = 0, 0, list(), None

    if job_status is None:
        return job_status, total_log_nums, total_log_pages, log_line_list, job_logs

    if not need_all_logs:
        job_logs = FedMLJobManager.get_instance().get_job_logs(job_id, page_num, page_size, api_key)

        if job_logs is not None:
            total_log_pages, total_log_nums = job_logs.total_num, job_logs.total_pages
            _parse_logs(log_line_list, job_logs)

        return job_status, total_log_nums, total_log_pages, log_line_list, job_logs

    job_logs = FedMLJobManager.get_instance().get_job_logs(job_id, 1, Constants.JOB_LOG_PAGE_SIZE, api_key)

    if job_logs is None:
        return job_status, total_log_nums, total_log_pages, log_line_list, job_logs

    # Get all logs
    if len(job_logs.log_lines):
        _parse_logs(log_line_list, job_logs)

        for page_count in range(2, job_logs.total_pages + 1):
            job_logs = FedMLJobManager.get_instance().get_job_logs(job_id, page_count,
                                                                   Constants.JOB_LOG_PAGE_SIZE, api_key)
            _parse_logs(log_line_list, job_logs)

    return job_status, job_logs.total_num, job_logs.total_pages, log_line_list, job_logs


def _authenticated_and_validated_platform(api_key, platform):
    authenticate(api_key)

    if not platform_is_valid(platform):
        return False

    return True


def _parse_logs(log_line_list, job_logs):
    if len(job_logs.log_lines):
        for log_line in job_logs.log_lines:
            log_line_list.append(log_line)
