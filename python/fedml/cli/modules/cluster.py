import click

import fedml.api

from prettytable import PrettyTable


# Message strings constants
confirmation_message: str = "Are you sure you want to {} these clusters?"
failure_message: str = ("Failed to {} the clusters, please check the arguments are valid and your network "
                        "connection and make sure be able to access the FedML® Launch platform.")
version_help: str = "specify version of MLOps platform. It should be dev, test or release"
api_key_help: str = "user api key."
cluster_action_help: str = "{} clusters from the MLOps platform."


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


@fedml_clusters.command("start", help=cluster_action_help.format("Start"))
@click.help_option("--help", "-h")
@click.argument("cluster_names", nargs=-1, callback=validate_cluster_names)
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def start(version, api_key, cluster_names):
    is_started = fedml.api.cluster_start(version=version, api_key=api_key, cluster_names=cluster_names)
    if is_started:
        click.echo("Clusters have been started.")
    else:
        click.echo(failure_message.format("start"))


@fedml_clusters.command("startall", help=cluster_action_help.format("Start ALL"))
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def startall(version, api_key):
    cluster_list_obj = fedml.api.cluster_list(version=version, api_key=api_key)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
        if click.confirm(confirmation_message.format("start"), abort=False):
            is_started = fedml.api.cluster_startall(version=version, api_key=api_key)
            if is_started:
                click.echo("Clusters have been started.")
            else:
                click.echo(failure_message.format("start"))
    else:
        click.echo("No clusters found.")


@fedml_clusters.command("stop", help=cluster_action_help.format("Stop"))
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
    help=version_help,
)
def stop(version, api_key, cluster_names):
    is_stopped = fedml.api.cluster_stop(version=version, api_key=api_key, cluster_names=cluster_names)
    if is_stopped:
        click.echo("Clusters have been stopped.")
    else:
        click.echo(failure_message.format("stop"))


@fedml_clusters.command("stopall", help=cluster_action_help.format("Stop ALL"))
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def stopall(version, api_key):
    cluster_list_obj = fedml.api.cluster_list(version=version, api_key=api_key)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
        if click.confirm(confirmation_message.format("stop"), abort=False):
            is_stopped = fedml.api.cluster_stopall(version=version, api_key=api_key)
            if is_stopped:
                click.echo("Clusters have been stopped.")
            else:
                click.echo(failure_message.format("stop"))
    else:
        click.echo("No clusters found.")


@fedml_clusters.command("kill", help=cluster_action_help.format("Kill (tear down)"))
@click.help_option("--help", "-h")
@click.argument("cluster_names", nargs=-1, callback=validate_cluster_names)
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def kill(version, api_key, cluster_names):
    is_killed = fedml.api.cluster_kill(version=version, api_key=api_key, cluster_names=cluster_names)
    if is_killed:
        click.echo("Clusters have been killed.")
    else:
        click.echo(failure_message.format("kill"))


@fedml_clusters.command("killall", help=cluster_action_help.format("Kill (tear down) ALL"))
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def killall(version, api_key):
    cluster_list_obj = fedml.api.cluster_list(version=version, api_key=api_key)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
        if click.confirm(confirmation_message.format("kill"), abort=False):
            is_killed = fedml.api.cluster_killall(version=version, api_key=api_key)
            if is_killed:
                click.echo("Clusters have been killed.")
            else:
                click.echo(failure_message.format("kill"))

    else:
        click.echo("No clusters found.")


@fedml_clusters.command("list", help=cluster_action_help.format("List"))
@click.help_option("--help", "-h")
@click.argument("cluster_names", nargs=-1)
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def list_clusters(version, api_key, cluster_names):
    cluster_list_obj = fedml.api.cluster_list(version=version, api_key=api_key, cluster_names=cluster_names)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
    else:
        click.echo("No clusters found.")


def _print_clusters(cluster_list_obj):
    click.echo("Found the following matching clusters.")
    cluster_list_table = PrettyTable(['Cluster Name', 'Cluster ID', 'Status'])

    for cluster in cluster_list_obj.cluster_list:
        cluster_list_table.add_row([cluster.cluster_name, cluster.cluster_id, cluster.status])

    print(cluster_list_table)
