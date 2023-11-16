from typing import List

from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterManager, FedMLClusterModelList
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLGpuDevices


def start(cluster_names, api_key) -> bool:
    authenticate(api_key)
    is_started = FedMLClusterManager.get_instance().start_clusters(cluster_names)
    return is_started


def stop(cluster_names, api_key) -> bool:
    authenticate(api_key)
    is_stopped = FedMLClusterManager.get_instance().stop_clusters(cluster_names)
    return is_stopped


def autostop(cluster_id, time, api_key) -> bool:
    authenticate(api_key)
    is_configured = FedMLClusterManager.get_instance().autostop_clusters(cluster_id, time)
    return is_configured


def kill(cluster_names, api_key) -> bool:
    authenticate(api_key)
    is_killed = FedMLClusterManager.get_instance().kill_clusters(cluster_names)
    return is_killed


def list_clusters(cluster_names, api_key) -> FedMLClusterModelList:
    authenticate(api_key)
    cluster_list_obj = FedMLClusterManager.get_instance().list_clusters(cluster_names)
    return cluster_list_obj


def status(cluster_name, api_key) -> (str, FedMLClusterModelList):
    authenticate(api_key)
    cluster_status_obj = FedMLClusterManager.get_instance().list_clusters(cluster_name)
    if cluster_status_obj is None or not len(cluster_status_obj.cluster_list):
        return None, None
    if cluster_status_obj is not None:
        if len(cluster_status_obj.cluster_list) > 1:
            raise Exception("More than one cluster found with the same name.")
    return cluster_status_obj.cluster_list[0].status, cluster_status_obj


def exists(cluster_name, api_key) -> bool:
    authenticate(api_key)
    cluster_list_obj = FedMLClusterManager.get_instance().list_clusters()
    if cluster_list_obj is None or not len(cluster_list_obj.cluster_list):
        return False
    clusters = set(map(lambda x: x.cluster_name, cluster_list_obj.cluster_list))
    return cluster_name in clusters


def confirm_and_start(run_id: str, cluster_id: str, gpu_matched: List[FedMLGpuDevices], api_key: str = None):
    authenticate(api_key)
    return FedMLClusterManager.get_instance().confirm_and_start(run_id=run_id, cluster_id=cluster_id,
                                                                gpu_matched=gpu_matched)
