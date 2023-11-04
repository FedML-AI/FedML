from typing import List
from fedml.launch import internals
from fedml.computing.scheduler.scheduler_entry.run_manager import FeatureEntryPoint

__all__ = [
    "job",
    "job_on_cluster",
]


def job(
        yaml_file: str, api_key: str = None, resource_id: str = None, device_server: str = None,
        device_edges: List[str] = None,
        feature_entry_point: FeatureEntryPoint = FeatureEntryPoint.FEATURE_ENTRYPOINT_API) -> internals.LaunchResult:
    """
    Launch a job on the FedML AI Nexus platform

    Args:
        yaml_file: Full path of your job yaml file.
        api_key: Your API key (if not configured already). (Default value = None)
        resource_id:
            Specific `resource_id` to use. Typically, you won't need to specify a specific `resource_id`.
            Instead, we will match resources based on your job yaml, and then automatically launch the job
            using matched resources.
        device_server:
            `device_server` to use. Only needed when you want to launch a federated learning job with specific
            `device_server` and `device_edges`
        device_edges:
            List of `device_edges` to use. Only needed when you want to launch a federated learning job
            with specific `device_server` and `device_edges`
        feature_entry_point:
            Entry point where you launch a job. Default entry point is from API.

    Returns:
        LaunchResult object with the following attributes

            result_code:
                API result code. `0` means success.
            result_msg:
                API status message.
            run_id:
                Run ID of the launched job.
            project_id:
                Project Id of the launched job. This is default assigned if not specified in your job yaml file
            inner_id:
                Serving endpoint id of launched job. Only applicable for Deploy / Serve Job tasks,
                and will be `None` otherwise.
    """

    return internals.job(yaml_file, api_key, resource_id, device_server, device_edges,
                         feature_entry_point=feature_entry_point)


def job_on_cluster(
        yaml_file: str, cluster: str, api_key: str = None, resource_id: str = None,
        device_server: str = None, device_edges: List[str] = None,
        feature_entry_point: FeatureEntryPoint = FeatureEntryPoint.FEATURE_ENTRYPOINT_API) -> internals.LaunchResult:
    """
    Launch a job on a cluster on the FedML AI Nexus platform

    Args:
        yaml_file: Full path of your job yaml file.
        cluster: Cluster name to use. If a cluster with provided name doesn't exist, one will be created.
        api_key: Your API key (if not configured already).
        resource_id: Specific `resource_id` to use. Typically, you won't need to specify a specific `resource_id`. Instead, we will match resources based on your job yaml, and then automatically launch the job using matched resources.
        device_server: `device_server` to use. Only needed when you want to launch a federated learning job with specific `device_server` and `device_edges`
        device_edges: List of `device_edges` to use. Only needed when you want to launch a federated learning job with specific `device_server` and `device_edges`
        feature_entry_point: Entry point where you launch a job. Default entry point is from API.
    Returns:
        LaunchResult object with the following attributes

            result_code:
                API result code. `0` means success.
            result_msg:
                API status message.
            run_id:
                Run ID of the launched job.
            project_id:
                Project Id of the launched job. This is default assigned if not specified in your job yaml file
            inner_id:
                Serving endpoint id of launched job. Only applicable for Deploy / Serve Job tasks,
                and will be `None` otherwise.
    """

    return internals.job_on_cluster(yaml_file=yaml_file, cluster=cluster, api_key=api_key, resource_id=resource_id,
                                    device_server=device_server, device_edges=device_edges,
                                    feature_entry_point=feature_entry_point)
