import click

from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterManager

from prettytable import PrettyTable


def kill(version, api_key, cluster_names, show_hint_texts):
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)

    is_stopped = FedMLClusterManager.get_instance().kill_clusters(cluster_names)

    if show_hint_texts:
        if is_stopped:
            click.echo("Clusters have been stopped.")
        else:
            click.echo("Failed to stop the clusters, please check the arguments are valid and your network connection "
                       "and make sure be able to access the FedML® Launch platform.")

    return is_stopped


def killall(version, api_key, show_hint_texts):
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    cluster_list_obj = FedMLClusterManager.get_instance().list_clusters()
    if cluster_list_obj is not None and len(cluster_list_obj.cluster_list) > 0:
        print_clusters(cluster_list_obj)
        if click.confirm(f"Are you sure you want to kill all these clusters?", abort=False):
            kill(version=version, api_key=api_key, cluster_names=(), show_hint_texts=show_hint_texts)
    else:
        click.echo("No clusters found.")


def list_clusters(version, api_key, cluster_names):
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    cluster_list_obj = FedMLClusterManager.get_instance().list_clusters(cluster_names)
    if cluster_list_obj is not None and len(cluster_list_obj.cluster_list) > 0:
        print_clusters(cluster_list_obj)
    else:
        click.echo("No clusters found.")


def print_clusters(cluster_list_obj):
    click.echo("Found the following matching clusters.")
    cluster_list_table = PrettyTable(['Cluster Name', 'Cluster ID', 'Status'])

    for cluster in cluster_list_obj.cluster_list:
        cluster_list_table.add_row([cluster.cluster_name, cluster.cluster_id, cluster.status])

    print(cluster_list_table)
