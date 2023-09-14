import click

from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobManager


def stop_jobs_core(platform, job_id, api_key, version, show_hint_texts=True):
    if not platform_is_valid(platform):
        return

    FedMLJobManager.get_instance().set_config_version(version)
    is_stopped = FedMLJobManager.get_instance().stop_job(platform, job_id, api_key)
    if show_hint_texts:
        if is_stopped:
            click.echo("Job has been stopped.")
        else:
            click.echo("Failed to stop the job, please check your network connection "
                       "and make sure be able to access the MLOps platform.")
    return is_stopped
