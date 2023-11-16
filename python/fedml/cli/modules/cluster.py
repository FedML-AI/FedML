import click

import fedml.api

from prettytable import PrettyTable

# Message strings constants
confirmation_message: str = "Are you sure you want to {} these clusters?"
failure_message: str = ("Failed to {} the clusters, please check the arguments are valid and your network "
                        "connection and make sure be able to access the FedML® Nexus AI Platform.")
version_help: str = "specify version of FedML® Nexus AI Platform. It should be dev, test or release"
api_key_help: str = "user api key."
cluster_action_help: str = "{} clusters from FedML® Nexus AI Platform"


@click.group("cluster")
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
def fedml_clusters(api_key, version):
    """
    Manage clusters on FedML® Nexus AI Platform
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
def start(cluster_names, version, api_key):
    fedml.set_env_version(version)
    is_started = fedml.api.cluster_start(api_key=api_key, cluster_names=cluster_names)
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
    fedml.set_env_version(version)
    cluster_list_obj = fedml.api.cluster_list(api_key=api_key)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
        if click.confirm(confirmation_message.format("start"), abort=False):
            is_started = fedml.api.cluster_startall(api_key=api_key)
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
def stop(cluster_names, version, api_key):
    fedml.set_env_version(version)
    is_stopped = fedml.api.cluster_stop(api_key=api_key, cluster_names=cluster_names)
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
    fedml.set_env_version(version)
    cluster_list_obj = fedml.api.cluster_list(api_key=api_key)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
        if click.confirm(confirmation_message.format("stop"), abort=False):
            is_stopped = fedml.api.cluster_stopall(api_key=api_key)
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
def kill(cluster_names, version, api_key):
    fedml.set_env_version(version)
    is_killed = fedml.api.cluster_kill(api_key=api_key, cluster_names=cluster_names)
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
    fedml.set_env_version(version)
    cluster_list_obj = fedml.api.cluster_list(api_key=api_key)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
        if click.confirm(confirmation_message.format("kill"), abort=False):
            is_killed = fedml.api.cluster_killall(api_key=api_key)
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
def list_clusters(cluster_names, version, api_key):
    fedml.set_env_version(version)
    cluster_list_obj = fedml.api.cluster_list(api_key=api_key, cluster_names=cluster_names)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
    else:
        click.echo("No clusters found.")


@fedml_clusters.command("status", help=cluster_action_help.format("Status of"))
@click.help_option("--help", "-h")
@click.argument("cluster_name", nargs=1)
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
def status(cluster_name, version, api_key):
    fedml.set_env_version(version)
    cluster_status, cluster_list_obj = fedml.api.cluster_status(cluster_name=(cluster_name,), api_key=api_key)
    if cluster_status is None:
        click.echo("No cluster found with the given name.")
    else:
        _print_clusters(cluster_list_obj)


@fedml_clusters.command("autostop", help="Autostop clusters after some minutes of inactivity. Defaults to 10 minutes")
@click.help_option("--help", "-h")
@click.argument("cluster_id", type=int,nargs=1)
@click.option("--time", "-t", type=int, default=10, help="Number of minutes of inactivity before autostop")
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
def autostop(cluster_id, time, version, api_key):
    fedml.set_env_version(version)
    is_configured = fedml.api.cluster_autostop(cluster_id=cluster_id, time=time, api_key=api_key)
    if is_configured:
        click.echo(f"Cluster has been successfully configured to autostop after {minutes} of inactivity.")
    else:
        click.echo("Cluster autostop configuration failed. The cluster will still undergo autostopping after the "
                   "default minutes of inactivity or the previously configured minutes, if any.")


def _print_clusters(cluster_list_obj):
    click.echo("Found the following matching clusters.")
    cluster_list_table = PrettyTable(['Cluster Name', 'Cluster ID', 'Status'])

    for cluster in cluster_list_obj.cluster_list:
        cluster_list_table.add_row([cluster.cluster_name, cluster.cluster_id, cluster.status])

    click.echo(cluster_list_table)
