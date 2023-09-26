from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterManager, FedMLClusterModelList


def kill(version, api_key, cluster_names, show_hint_texts):
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)

    is_stopped = FedMLClusterManager.get_instance().kill_clusters(cluster_names)

    if show_hint_texts:
        if is_stopped:
            print("Clusters have been stopped.")
        else:
            print("Failed to stop the clusters, please check the arguments are valid and your network connection "
                       "and make sure be able to access the FedML® Launch platform.")

    return is_stopped


def killall(version, api_key, show_hint_texts):
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    kill(version=version, api_key=api_key, cluster_names=(), show_hint_texts=show_hint_texts)


def list_clusters(version, api_key, cluster_names) -> FedMLClusterModelList:
    authenticate(api_key, version)
    FedMLClusterManager.get_instance().set_config_version(version)
    cluster_list_obj = FedMLClusterManager.get_instance().list_clusters(cluster_names)
    return cluster_list_obj
