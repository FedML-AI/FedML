import click

from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobManager
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager
from .utils import stop_jobs_core


@click.group("jobs")
def jobs():
    """
    Manage jobs on the MLOps platform.
    """
    pass


@jobs.command("start", help="Start a job at the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="octopus",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch."
)
@click.option(
    "--project_name",
    "-prj",
    type=str,
    help="The project name at the MLOps platform.",
)
@click.option(
    "--application_name",
    "-app",
    type=str,
    help="Application name in the My Application list at the MLOps platform.",
)
@click.option(
    "--job_name",
    "-jn",
    type=str,
    default="",
    help="The job name at the MLOps platform.",
)
@click.option(
    "--devices_server", "-ds", type=str, default="",
    help="The server to run the launching job, for the launch platform, we do not need to set this option."
)
@click.option(
    "--devices_edges", "-de", type=str, default="",
    help="The edge devices to run the launching job. Seperated with ',', e.g. 705,704. "
         "for the launch platform, we do not need to set this option."
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="start job at which version of MLOps platform. It should be dev, test or release",
)
def start_job(platform, project_name, application_name, job_name, devices_server, devices_edges, user, api_key,
              version):
    if not platform_is_valid(platform):
        return

    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLJobManager.get_instance().set_config_version(version)
    result = FedMLJobManager.get_instance().start_job(platform, project_name, application_name,
                                                      devices_server, devices_edges,
                                                      user, api_key,
                                                      job_name=job_name, need_confirmation=False)
    if result:
        click.echo(f"Job {result.job_name} has started.")
        click.echo(f"Please go to this web page with your account id {result.user_id} to review your job details.")
        click.echo(f"{result.job_url}")
        click.echo(f"For querying the status of the job, please run the command: "
                   f"fedml jobs list -id {result.job_id}")
    else:
        click.echo("Failed to start job, please check your network connection "
                   "and make sure be able to access the MLOps platform.")


@jobs.command("list", help="List jobs from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--job_id",
    "-id",
    type=str,
    default="",
    help="Job id at the MLOps platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="list jobs at which version of MLOps platform. It should be dev, test or release",
)
def list_jobs(platform, job_id, api_key, version):
    if not platform_is_valid(platform):
        return

    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().list_jobs(job_id)


@jobs.command("stop", help="Stop a job from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--job_id",
    "-id",
    type=str,
    default="",
    help="Job id at the MLOps platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="stop a job at which version of MLOps platform. It should be dev, test or release",
)
def stop_jobs(platform, job_id, api_key, version):
    stop_jobs_core(platform, job_id, api_key, version)
