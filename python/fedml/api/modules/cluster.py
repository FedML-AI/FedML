from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterManager, FedMLClusterModelList


def start(cluster_names, version, api_key) -> bool:
    authenticate(api_key, version)
    is_started = FedMLClusterManager.get_instance().start_clusters(cluster_names)
    return is_started


def stop(cluster_names, version, api_key) -> bool:
    authenticate(api_key, version)
    is_stopped = FedMLClusterManager.get_instance().stop_clusters(cluster_names)
    return is_stopped


def kill(cluster_names, version, api_key) -> bool:
    authenticate(api_key, version)
    is_killed = FedMLClusterManager.get_instance().kill_clusters(cluster_names)
    return is_killed


def list_clusters(cluster_names, version, api_key) -> FedMLClusterModelList:
    authenticate(api_key, version)
    cluster_list_obj = FedMLClusterManager.get_instance().list_clusters(cluster_names)
    return cluster_list_obj


def status(cluster_name, version, api_key) -> (str, FedMLClusterModelList):
    authenticate(api_key, version)
    cluster_status_obj = FedMLClusterManager.get_instance().list_clusters(cluster_name)
    if cluster_status_obj is None or not len(cluster_status_obj.cluster_list):
        return None, None
    if cluster_status_obj is not None:
        if len(cluster_status_obj.cluster_list) > 1:
            raise Exception("More than one cluster found with the same name.")
    return cluster_status_obj.cluster_list[0].status, cluster_status_obj
