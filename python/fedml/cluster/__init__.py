from typing import Tuple
from fedml.cluster import internals


def list(cluster_names: Tuple[str] = (), api_key: str = None) -> internals.FedMLClusterModelList:
    return internals.list_clusters(cluster_names=cluster_names, api_key=api_key)


def exists(cluster_name: str, api_key: str = None) -> bool:
    return internals.exists(cluster_name=cluster_name, api_key=api_key)


def status(cluster_name: Tuple[str], api_key: str = None) -> (str, internals.FedMLClusterModelList):
    return internals.status(cluster_name=cluster_name, api_key=api_key)


def start(cluster_names: Tuple[str], api_key: str = None) -> bool:
    return internals.start(cluster_names=cluster_names, api_key=api_key)


def startall(api_key: str = None) -> bool:
    return internals.start(cluster_names=(), api_key=api_key)


def stop(cluster_names: Tuple[str], api_key: str = None) -> bool:
    return internals.stop(cluster_names=cluster_names, api_key=api_key)


def stopall(api_key: str = None) -> bool:
    return internals.stop(cluster_names=(), api_key=api_key)


def kill(cluster_names: Tuple[str], api_key: str = None) -> bool:
    return internals.kill(cluster_names=cluster_names, api_key=api_key)


def killall(api_key=None) -> bool:
    return internals.kill(cluster_names=(), api_key=api_key)
