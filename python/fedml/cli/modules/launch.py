import click

import fedml.api
from fedml.cli.modules.utils import DefaultCommandGroup


@click.group("launch", cls=DefaultCommandGroup, default_command='default')
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
    fedml.api.launch_job(yaml_file[0], version=version, api_key=api_key)


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
    fedml.api.stop_job(job_id[0], version, platform, api_key)


@fedml_launch.command("log", help="View the job list at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("job_id", nargs=-1)
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
def fedml_launch_log(job_id, api_key, version):
    fedml.api.launch_log(job_id[0], 0, 0, version, api_key, need_all_logs=True)


@fedml_launch.command("queue", help="View the job queue at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("group_id", nargs=-1)
def fedml_launch_queue(group_id):
    pass