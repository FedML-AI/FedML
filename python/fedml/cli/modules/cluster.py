import click

import fedml.api


@click.group("cluster")
@click.help_option("--help", "-h")
def fedml_clusters():
    """
    Manage clusters on the MLOps platform.
    """
    pass


# Callback function to validate cluster_names
def validate_cluster_names(ctx, param, value):
    if not value:
        raise click.BadParameter("At least one cluster name must be provided.")
    return value


@fedml_clusters.command("kill", help="Kill (tear down) clusters from the MLOps platform.")
@click.help_option("--help", "-h")
@click.argument("cluster_names", nargs=-1, callback=validate_cluster_names)
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
def kill(version, api_key, cluster_names):
    fedml.api.kill_clusters(version=version, api_key=api_key, cluster_names=cluster_names)


@fedml_clusters.command("killall", help="Kill (tear down) ALL clusters from the MLOps platform.")
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
def killall(version, api_key):
    fedml.api.killall_clusters(version=version, api_key=api_key)


@fedml_clusters.command("list", help="List clusters from the MLOps platform.")
@click.help_option("--help", "-h")
@click.argument("cluster_names", nargs=-1)
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
def list_clusters(version, api_key, cluster_names):
    fedml.api.list_clusters(version=version, api_key=api_key, cluster_names=cluster_names)
