from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterManager, FedMLClusterModelList


def start(version, api_key, cluster_names) -> bool:
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    is_started = FedMLClusterManager.get_instance().start_clusters(cluster_names)
    return is_started


def stop(version, api_key, cluster_names) -> bool:
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    is_stopped = FedMLClusterManager.get_instance().stop_clusters(cluster_names)
    return is_stopped


def kill(version, api_key, cluster_names) -> bool:
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    is_killed = FedMLClusterManager.get_instance().kill_clusters(cluster_names)
    return is_killed


def list_clusters(version, api_key, cluster_names) -> FedMLClusterModelList:
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    cluster_list_obj = FedMLClusterManager.get_instance().list_clusters(cluster_names)
    return cluster_list_obj


