import click

from fedml.cli.modules import utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobManager
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager


@click.group("launch", cls=utils.DefaultCommandGroup, default_command='default')
@click.help_option("--help", "-h")
def fedml_launch():
    """
    Manage resources on the FedML® Launch platform (open.fedml.ai).
    """


@fedml_launch.command(
    "default", help="Launch job at the FedML® Launch platform (open.fedml.ai)",
    context_settings={"ignore_unknown_options": True}
)
@click.help_option("--help", "-h")
@click.argument("yaml_file", nargs=-1)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--group",
    "-g",
    type=str,
    default="",
    help="The queue group id on which your job will be scheduled.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="launch job to which version of MLOps platform. It should be dev, test or release",
)
@click.option(
    "--cluster",
    "-c",
    type=str,
    help="Please provide a cluster name. If a cluster with that name already exists, it will be used; otherwise, "
         "a new cluster with the provided name will be created."
)
def fedml_launch_default(yaml_file, api_key, group, cluster, version):
    """
    Manage resources on the FedML® Launch platform (open.fedml.ai).
    """
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().api_launch_job(yaml_file[0], None)


@fedml_launch.command("cancel", help="Cancel job at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("job_id", nargs=-1)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="stop a job at which version of FedML® Launch platform. It should be dev, test or release",
)
def fedml_launch_cancel(job_id, platform, api_key, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    api_key = FedMLLaunchManager.get_api_key()
    utils.stop_job_wrapper(platform, job_id[0], api_key, version)


@fedml_launch.command("log", help="View the job list at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("job_id", nargs=-1)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="list jobs at which version of the FedML® Launch platform. It should be dev, test or release",
)
def fedml_launch_log(job_id, platform, api_key, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().api_launch_log(job_id[0], 0, 0, need_all_logs=True)


@fedml_launch.command("queue", help="View the job queue at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("group_id", nargs=-1)
def fedml_launch_queue(group_id):
    pass


@fedml_launch.group("utils", context_settings={"ignore_unknown_options": True}, invoke_without_command=True)
@click.help_option("--help", "-h")
def fedml_launch_utils():
    """
    Manage launch related utils on the MLOps platform.
    """
    pass


@fedml_launch_utils.command(
    "launch-octopus", help="Launch job at the FedML® Launch platform (open.fedml.ai)",
    context_settings={"ignore_unknown_options": True}
)
@click.help_option("--help", "-h")
@click.argument("yaml_file", nargs=-1)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--group",
    "-g",
    type=str,
    default="",
    help="The queue group id on which your job will be scheduled.",
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
    "--version",
    "-v",
    type=str,
    default="release",
    help="launch job to which version of MLOps platform. It should be dev, test or release",
)
def fedml_launch_utils_launch_octopus(yaml_file, api_key, group, devices_server, devices_edges, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().platform_type = SchedulerConstants.PLATFORM_TYPE_OCTOPUS
    FedMLLaunchManager.get_instance().device_server = devices_server
    FedMLLaunchManager.get_instance().device_edges = devices_edges
    FedMLLaunchManager.get_instance().api_launch_job(yaml_file[0], None)


@fedml_launch_utils.command("start-job", help="Start a job at the MLOps platform.")
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
def fedml_launch_utils_start_job(platform, project_name, application_name, job_name, devices_server, devices_edges, user, api_key,
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


@fedml_launch_utils.command("list-job", help="List jobs from the MLOps platform.")
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
def fedml_launch_utils_list_jobs(platform, job_id, api_key, version):
    if not platform_is_valid(platform):
        return

    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().list_jobs(job_id)


@fedml_launch_utils.command("stop-job", help="Stop a job from the MLOps platform.")
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
def fedml_launch_utils_stop_job(platform, job_id, api_key, version):
    utils.stop_job_wrapper(platform, job_id, api_key, version)

