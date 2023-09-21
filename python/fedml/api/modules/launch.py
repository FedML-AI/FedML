from fedml.api.modules.utils import authenticate

from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager


def job(yaml_file, api_key, version, resource_id, prompt=True):
    authenticate(api_key, version)
    FedMLLaunchManager.get_instance().set_config_version(version)
    return FedMLLaunchManager.get_instance().api_launch_job(yaml_file, resource_id=resource_id, prompt=prompt)


def log(job_id, version, api_key, page_num, page_size, need_all_logs):
    authenticate(api_key, version)
    return FedMLLaunchManager.get_instance().api_launch_log(job_id, page_num, page_size, need_all_logs)

