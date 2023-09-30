from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager, FedMLJobManager


def stop(job_id, version, platform, api_key):
    authenticate(api_key, version)

    if not platform_is_valid(platform):
        return

    FedMLJobManager.get_instance().set_config_version(version)
    is_stopped = FedMLJobManager.get_instance().stop_job(platform, api_key, job_id)
    return is_stopped


def list_jobs(version, job_name, job_id, platform, api_key):
    authenticate(api_key, version)

    if not platform_is_valid(platform):
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().list_jobs(job_name, job_id)
