import click

from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager, FedMLJobManager


def stop(job_id, version, platform, api_key, show_hint_texts):
    authenticate(api_key, version)

    if not platform_is_valid(platform):
        return

    FedMLJobManager.get_instance().set_config_version(version)
    is_stopped = FedMLJobManager.get_instance().stop_job(platform, api_key, job_id)

    if show_hint_texts:
        if is_stopped:
            click.echo("Job has been stopped.")
        else:
            click.echo("Failed to stop the job, please check the arguments are valid and your network connection "
                       "and make sure be able to access the FedML® Launch platform.")

    return is_stopped


def lists(version, job_name, job_id, platform, api_key):
    authenticate(api_key, version)

    if not platform_is_valid(platform):
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().list_jobs(job_name, job_id)
