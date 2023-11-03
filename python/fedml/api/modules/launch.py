import os
from typing import List

from fedml.api.modules.utils import authenticate
from fedml.api.modules.run import create, create_on_cluster, start
from fedml.api.modules.cluster import confirm_and_start

from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager, FedMLAppManager
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunStartedModel, FeatureEntryPoint

from fedml.api.constants import ApiConstants
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.scheduler_entry.constants import Constants

from fedml.computing.scheduler.comm_utils.security_utils import get_api_key


class LaunchResult:
    def __init__(self, result_code: int, result_message: str, run_id: str = None, project_id: str = None,
                 inner_id: str = None):
        self.run_id = run_id
        self.project_id = project_id
        self.inner_id = inner_id
        self.result_code = result_code
        self.result_message = result_message


def create_run(yaml_file, api_key: str, resource_id: str = None, device_server: str = None,
               device_edges: List[str] = None, feature_entry_point: FeatureEntryPoint = None) -> (int, str, FedMLRunStartedModel):
    result_code, result_message = (ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED],
                                   ApiConstants.LAUNCHED)

    authenticate(api_key)

    # Check if resource is available.
    create_run_result = FedMLLaunchManager.get_instance().get_matched_result(resource_id)

    if create_run_result is None:
        # Prepare the application for launch.
        job_config, app_updated_result = _prepare_launch_app(yaml_file)

        if not app_updated_result:
            return ApiConstants.ERROR_CODE[
                ApiConstants.APP_UPDATE_FAILED], ApiConstants.APP_UPDATE_FAILED, create_run_result

        # Start the run with the above application.
        create_run_result = create(platform=SchedulerConstants.PLATFORM_TYPE_FALCON, job_config=job_config,
                                   device_server=device_server, device_edges=device_edges, api_key=get_api_key(),
                                   feature_entry_point=feature_entry_point)

        result_code, result_message = _parse_create_result(result=create_run_result, yaml_file=yaml_file)

        # TODO (alaydshah): Revisit if this is appropriate here or not.
        FedMLLaunchManager.get_instance().update_matched_result_if_gpu_matched(resource_id, create_run_result)

    return result_code, result_message, create_run_result


def create_run_on_cluster(yaml_file, cluster: str, api_key: str, resource_id: str = None, device_server: str = None,
                          device_edges: List[str] = None, feature_entry_point: FeatureEntryPoint = None) -> (int, str, FedMLRunStartedModel):
    result_code, result_message = (ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED],
                                   ApiConstants.LAUNCHED)

    authenticate(api_key)

    # Check if resource is available.
    create_run_result = FedMLLaunchManager.get_instance().get_matched_result(resource_id)

    if create_run_result is None:
        # Prepare the application for launch.
        job_config, app_updated_result = _prepare_launch_app(yaml_file)

        if not app_updated_result:
            return ApiConstants.ERROR_CODE[ApiConstants.APP_UPDATE_FAILED], ApiConstants.APP_UPDATE_FAILED, create_run_result

        # Start the job with the above application.
        create_run_result = create_on_cluster(platform=SchedulerConstants.PLATFORM_TYPE_FALCON,
                                              cluster=cluster, job_config=job_config, device_server=device_server,
                                              device_edges=device_edges, api_key=get_api_key(),
                                              feature_entry_point=feature_entry_point)

        result_code, result_message = _parse_create_result(result=create_run_result, yaml_file=yaml_file)

        # TODO (alaydshah): Revisit if this is appropriate here or not.
        FedMLLaunchManager.get_instance().update_matched_result_if_gpu_matched(resource_id=resource_id,
                                                                               result=create_run_result)

    return result_code, result_message, create_run_result


def run(create_run_result: FedMLRunStartedModel, api_key: str, device_server: str = None,
        device_edges: List[str] = None, feature_entry_point: FeatureEntryPoint = None):
    authenticate(api_key)

    # Start the run
    launch_result = start(platform=SchedulerConstants.PLATFORM_TYPE_FALCON, create_run_result=create_run_result,
                          device_server=device_server, device_edges=device_edges, api_key=get_api_key(),
                          feature_entry_point=feature_entry_point)

    return launch_result


def job(
        yaml_file, api_key: str, resource_id: str = None, device_server: str = None,
        device_edges: List[str] = None,
        feature_entry_point: FeatureEntryPoint = FeatureEntryPoint.FEATURE_ENTRYPOINT_API) -> LaunchResult:
    # Create Run
    result_code, result_message, create_run_result = create_run(yaml_file, api_key, resource_id, device_server,
                                                                device_edges, feature_entry_point=feature_entry_point)

    if not create_run_result:
        return LaunchResult(result_code=result_code, result_message=result_message)

    run_id = getattr(create_run_result, "run_id", None)
    project_id = getattr(create_run_result, "project_id", None)

    inner_id = run_id if create_run_result.inner_id is None else create_run_result.inner_id

    if ((result_code == ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED] and not create_run_result.user_check) or
            (result_code != ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED])):
        return LaunchResult(result_code=result_code, result_message=result_message, run_id=run_id,
                            project_id=project_id, inner_id=inner_id)

    # Run Job
    run_result = run(create_run_result=create_run_result, api_key=api_key, device_server=device_server,
                     device_edges=device_edges, feature_entry_point=feature_entry_point)

    # Return Result
    if run_result is None:
        return LaunchResult(result_code=ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED],
                            result_message=ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED, run_id=run_id,
                            project_id=project_id, inner_id=inner_id)

    if run_result.run_url == "":
        return LaunchResult(result_code=ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_JOB_URL_ERROR],
                            result_message=ApiConstants.LAUNCH_JOB_STATUS_JOB_URL_ERROR, run_id=run_id,
                            project_id=project_id, inner_id=inner_id)

    return LaunchResult(result_code=ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED],
                        result_message=ApiConstants.LAUNCHED,
                        run_id=run_id, project_id=project_id, inner_id=inner_id)


def job_on_cluster(yaml_file, cluster: str, api_key: str, resource_id: str, device_server: str,
                   device_edges: List[str],
                   feature_entry_point: FeatureEntryPoint = FeatureEntryPoint.FEATURE_ENTRYPOINT_API) -> LaunchResult:
    # Schedule Job
    result_code, result_message, create_run_result = create_run_on_cluster(
        yaml_file, cluster, api_key, resource_id, device_server, device_edges, feature_entry_point=feature_entry_point)

    if not create_run_result:
        return LaunchResult(result_code=result_code, result_message=result_message)

    run_id = getattr(create_run_result, "run_id", None)
    project_id = getattr(create_run_result, "project_id", None)
    inner_id = run_id if create_run_result.inner_id is None else create_run_result.inner_id

    # Return if run launched and no user check required, or launch failed
    if ((result_code == ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED] and not create_run_result.user_check) or
            (result_code != ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED])):
        return LaunchResult(result_code=result_code, result_message=result_message, run_id=run_id,
                            project_id=project_id, inner_id=inner_id)

    cluster_id = getattr(create_run_result, "cluster_id", None)

    if cluster_id is None or cluster_id == "":
        return LaunchResult(result_code=ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CREATION_FAILED],
                            result_message=ApiConstants.CLUSTER_CREATION_FAILED,
                            run_id=run_id, project_id=project_id, inner_id=inner_id)

    # Confirm cluster and start job
    cluster_confirmed = confirm_and_start(run_id=run_id, cluster_id=cluster_id,
                                          gpu_matched=create_run_result.gpu_matched)

    if cluster_confirmed:
        return LaunchResult(result_code=ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CONFIRM_SUCCESS],
                            result_message=ApiConstants.CLUSTER_CONFIRM_SUCCESS, run_id=run_id, project_id=project_id,
                            inner_id=inner_id)

    return LaunchResult(result_code=ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CONFIRM_FAILED],
                        result_message=ApiConstants.CLUSTER_CONFIRM_FAILED, run_id=run_id, project_id=project_id,
                        inner_id=inner_id)


def _prepare_launch_app(yaml_file):
    job_config, app_config, client_package, server_package = FedMLLaunchManager.get_instance().prepare_launch(
        yaml_file)

    # Create and update an application with the built packages.
    app_updated_result = FedMLAppManager.get_instance().update_app(
        SchedulerConstants.PLATFORM_TYPE_FALCON, job_config.application_name, app_config, get_api_key(),
        client_package_file=client_package, server_package_file=server_package,
        workspace=job_config.workspace, model_name=job_config.serving_model_name,
        model_version=job_config.serving_model_version, model_url=job_config.serving_model_s3_url,
        app_id=job_config.job_id, config_id=job_config.config_id,
        job_type=Constants.JOB_TASK_TYPE_MAP.get(job_config.task_type, Constants.JOB_TASK_TYPE_TRAIN_CODE),
        job_subtype=Constants.JOB_TASK_SUBTYPE_MAP.get(job_config.task_subtype,
                                                       Constants.JOB_TASK_SUBTYPE_TRAIN_GENERAL_TRAINING_CODE))

    # Post processor to clean up local temporary launch package and do other things.
    FedMLLaunchManager.get_instance().post_launch(job_config)

    return job_config, app_updated_result


def _parse_create_result(result: FedMLRunStartedModel, yaml_file) -> (int, str):
    if not result:
        return (ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED],
                result.message)
    if result.status == Constants.JOB_START_STATUS_BIND_CREDIT_CARD_FIRST:
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_BIND_CREDIT_CARD_FIRST],
                ApiConstants.RESOURCE_MATCHED_STATUS_BIND_CREDIT_CARD_FIRST)
    if result.status == Constants.JOB_START_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED:
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED],
                ApiConstants.RESOURCE_MATCHED_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED)
    if result.status == Constants.JOB_START_STATUS_INVALID:
        return (ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_INVALID],
                f"\nPlease check your {os.path.basename(yaml_file)} file "
                f"to make sure the syntax is valid, e.g. "
                f"whether minimum_num_gpus or maximum_cost_per_hour is valid.")
    if result.status == Constants.JOB_START_STATUS_BLOCKED:
        return (ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_BLOCKED],
                ApiConstants.LAUNCH_JOB_STATUS_BLOCKED)
    if not result.run_url or result.run_url == "":
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_JOB_URL_ERROR],
                f"Failed to launch the job: "
                f"{result.message if result.message is not None else ApiConstants.RESOURCE_MATCHED_STATUS_JOB_URL_ERROR}")
    if result.status == Constants.JOB_START_STATUS_QUEUED:
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_QUEUED],
                ApiConstants.RESOURCE_MATCHED_STATUS_QUEUED)
    if result.status == Constants.JOB_START_STATUS_LAUNCHED:
        return (ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED],
                ApiConstants.LAUNCHED)
    else:
        return ApiConstants.ERROR_CODE[ApiConstants.ERROR], result.message

