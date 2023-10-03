import click

import fedml.api
from fedml.cli.modules.utils import DefaultCommandGroup


@click.group("launch", cls=DefaultCommandGroup, default_command='default')
@click.help_option("--help", "-h")
@click.option(
    "--cluster",
    "-c",
    default="",
    type=str,
    help="Please provide a cluster name. If a cluster with that name already exists, it will be used; otherwise, "
         "a new cluster with the provided name will be created."
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="launch job to which version of MLOps platform. It should be dev, test or release",
)
def fedml_launch(api_key, version, cluster):
    """
    Launch job at the FedML® Launch platform (open.fedml.ai).
    """
    pass


@fedml_launch.command(
    "default", help="Launch job at the FedML® Launch platform (open.fedml.ai)",
    context_settings={"ignore_unknown_options": True}, hidden=True
)
@click.help_option("--help", "-h")
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
    default="",
    type=str,
    help="Please provide a cluster name. If a cluster with that name already exists, it will be used; otherwise, "
         "a new cluster with the provided name will be created."
)
@click.argument("yaml_file", nargs=-1)
def fedml_launch_default(yaml_file, api_key, group, cluster, version):
    """
    Manage resources on the FedML® Launch platform (open.fedml.ai).
    """
    fedml.set_env_version(version)
    fedml.api.launch_job(yaml_file[0], cluster=cluster, api_key=api_key)
