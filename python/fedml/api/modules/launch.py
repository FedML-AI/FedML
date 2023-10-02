from fedml.api.modules.utils import authenticate

from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager


def job(yaml_file, api_key, resource_id, cluster="", prompt=True):
    authenticate(api_key)
    return FedMLLaunchManager.get_instance().api_launch_job(yaml_file, cluster=cluster, resource_id=resource_id, prompt=prompt)




