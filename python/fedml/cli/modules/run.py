import click
import os

import fedml.api
from fedml.cli.modules.utils import DefaultCommandGroup


def is_yaml_file(command):
    _, file_extension = os.path.splitext(command)
    return file_extension.lower() in {'.yaml', '.yml'}


@click.group("run", cls=DefaultCommandGroup, default_command="default")
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="specify version of MLOps platform. It should be dev, test or release",
)
def fedml_run(api_key, version):
    """
    Run commands into cluster \n
    Example Usage:
    fedml run <cluster_name> job.yaml
    """
    pass


@fedml_run.command("default", help="Run command in cluster", context_settings={"ignore_unknown_options": True},
                   hidden=True)
@click.help_option("--help", "-h")
@click.argument("cluster", nargs=1)
@click.argument("command", nargs=1)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="specify version of MLOps platform. It should be dev, test or release",
)
def run_default(command, cluster, api_key, version):
    fedml.set_env_version(version)
    if not is_yaml_file(command):
        raise click.BadParameter("Currently, it only supports running through yaml file. "
                                 "Future iterations will allow executing commands directly into cluster")
    if not fedml.api.cluster_exists(cluster, api_key):
        raise click.BadParameter(f"Cluster {cluster} does not exist. Run can only be executed on clusters that "
                                 f"already exists.")
    return fedml.api.run_command(command, cluster, api_key=api_key)
