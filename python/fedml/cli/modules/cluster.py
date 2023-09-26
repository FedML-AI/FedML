import click

import fedml.api

from prettytable import PrettyTable


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
    cluster_list_obj = fedml.api.list_clusters(version=version, api_key=api_key)
    if cluster_list_obj and cluster_list_obj.cluster_list:
        _print_clusters(cluster_list_obj)
        if click.confirm(f"Are you sure you want to kill all these clusters?", abort=False):
            fedml.api.killall_clusters(version=version, api_key=api_key)
    else:
        click.echo("No clusters found.")



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
    cluster_list_obj = fedml.api.list_clusters(version=version, api_key=api_key, cluster_names=cluster_names)
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
