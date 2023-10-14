import os

from fedml.api.modules.utils import authenticate
from fedml.api.modules.job import start, start_on_cluster, stop
from fedml.api.modules.cluster import confirm_and_start

from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager, FedMLAppManager

from fedml.api.constants import ApiConstants
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.scheduler_entry.constants import Constants

from fedml.computing.scheduler.comm_utils.security_utils import get_api_key


def schedule_job(yaml_file, api_key, resource_id, device_server, device_edges):
    result_code, result_message = (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED],
                                   ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED)

    authenticate(api_key)

    # Check if resource is available.
    schedule_result = FedMLLaunchManager.get_instance().get_matched_result(resource_id)

    if schedule_result is None:
        # Prepare the application for launch.
        job_config, app_updated_result = _prepare_launch_app(yaml_file)

        if not app_updated_result:
            return ApiConstants.ERROR_CODE[
                ApiConstants.APP_UPDATE_FAILED], ApiConstants.APP_UPDATE_FAILED, schedule_result

        # Start the job with the above application.
        schedule_result = start(SchedulerConstants.PLATFORM_TYPE_FALCON, job_config.project_name,
                                job_config.application_name,
                                device_server, device_edges, get_api_key(), no_confirmation=False,
                                model_name=job_config.serving_model_name,
                                model_endpoint=job_config.serving_endpoint_name,
                                job_yaml=job_config.job_config_dict, job_type=job_config.task_type,
                                app_job_id=job_config.job_id, app_job_name=job_config.job_name,
                                config_id=job_config.config_id)

        _post_process_launch_result(schedule_result, job_config)

        result_code, result_message = _parse_schedule_result(schedule_result, yaml_file)

        # TODO (alaydshah): Revisit if this is appropriate here or not.
        FedMLLaunchManager.get_instance().update_matched_result_if_gpu_matched(resource_id, schedule_result)

    return result_code, result_message, schedule_result


def schedule_job_on_cluster(yaml_file, cluster, api_key, resource_id, device_server, device_edges):
    result_code, result_message = (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED],
                                   ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED)

    authenticate(api_key)

    # Check if resource is available.
    schedule_result = FedMLLaunchManager.get_instance().get_matched_result(resource_id)

    if schedule_result is None:
        # Prepare the application for launch.
        job_config, app_updated_result = _prepare_launch_app(yaml_file)

        if not app_updated_result:
            return ApiConstants.ERROR_CODE[
                ApiConstants.APP_UPDATE_FAILED], ApiConstants.APP_UPDATE_FAILED, schedule_result

        # Start the job with the above application.
        schedule_result = start_on_cluster(SchedulerConstants.PLATFORM_TYPE_FALCON, cluster,
                                           job_config.project_name,
                                           job_config.application_name,
                                           device_server, device_edges, get_api_key(), no_confirmation=False,
                                           model_name=job_config.serving_model_name,
                                           model_endpoint=job_config.serving_endpoint_name,
                                           job_yaml=job_config.job_config_dict, job_type=job_config.task_type,
                                           app_job_id=job_config.job_id, app_job_name=job_config.job_name,
                                           config_id=job_config.config_id)

        _post_process_launch_result(schedule_result, job_config)

        result_code, result_message = _parse_schedule_result(schedule_result, yaml_file)

        # TODO (alaydshah): Revisit if this is appropriate here or not.
        FedMLLaunchManager.get_instance().update_matched_result_if_gpu_matched(resource_id, schedule_result)

    return result_code, result_message, schedule_result


def run_job(schedule_result, api_key, device_server, device_edges):
    authenticate(api_key)

    # Start the job
    launch_result = start(SchedulerConstants.PLATFORM_TYPE_FALCON,
                          schedule_result.project_name,
                          schedule_result.application_name,
                          device_server, device_edges, get_api_key(),
                          no_confirmation=True, job_id=schedule_result.job_id,
                          job_type=schedule_result.job_type,
                          app_job_id=schedule_result.app_job_id,
                          app_job_name=schedule_result.app_job_name)

    if launch_result is not None:
        launch_result.project_name = schedule_result.project_name
        launch_result.application_name = schedule_result.application_name

    return launch_result


def job(yaml_file, api_key, resource_id, device_server, device_edges):
    # Schedule Job
    result_code, result_message, schedule_result = schedule_job(yaml_file, api_key, resource_id, device_server,
                                                                device_edges)

    if schedule_result is None:
        return None, None, None, result_code, result_message

    job_id = getattr(schedule_result, "job_id", None)
    project_id = getattr(schedule_result, "project_id", None)

    inner_id = job_id if schedule_result.inner_id is None else schedule_result.inner_id

    if (result_code == ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_REQUEST_SUCCESS] or
            result_code != ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED]):
        return job_id, project_id, inner_id, result_code, result_message

    # Run Job
    run_result = run_job(schedule_result, api_key, device_server, device_edges)

    # Return Result
    if run_result is None:
        return job_id, project_id, inner_id, ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED], \
            ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED

    if run_result.job_url == "":
        return job_id, project_id, inner_id, ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_JOB_URL_ERROR], \
            ApiConstants.LAUNCH_JOB_STATUS_JOB_URL_ERROR

    return job_id, project_id, inner_id, 0, ""


def job_on_cluster(yaml_file, cluster, api_key, resource_id, device_server, device_edges):
    # Schedule Job
    result_code, result_message, schedule_result = schedule_job_on_cluster(yaml_file, cluster, api_key, resource_id,
                                                                           device_server, device_edges)

    if schedule_result is None:
        return None, None, None, result_code, result_message

    job_id = getattr(schedule_result, "job_id", None)
    project_id = getattr(schedule_result, "project_id", None)

    inner_id = job_id if schedule_result.inner_id is None else schedule_result.inner_id

    if (result_code == ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_REQUEST_SUCCESS] or
            result_code != ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED]):
        return job_id, project_id, inner_id, result_code, result_message

    cluster_id = getattr(schedule_result, "cluster_id", None)

    if cluster_id is None or cluster_id == "":
        return (job_id, project_id, inner_id, ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CREATION_FAILED],
                ApiConstants.CLUSTER_CREATION_FAILED)

    # Confirm cluster and start job
    cluster_confirmed = confirm_and_start(cluster_id, schedule_result.gpu_matched)

    if cluster_confirmed:
        return (job_id, project_id, inner_id, ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CONFIRM_SUCCESS],
                ApiConstants.CLUSTER_CONFIRM_SUCCESS)
    else:
        return (job_id, project_id, inner_id, ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CONFIRM_FAILED],
                ApiConstants.CLUSTER_CONFIRM_FAILED)


def _post_process_launch_result(launch_result, job_config):
    if launch_result is not None:
        launch_result.inner_id = job_config.serving_endpoint_id \
            if job_config.task_type == Constants.JOB_TASK_TYPE_DEPLOY or \
               job_config.task_type == Constants.JOB_TASK_TYPE_SERVE else None
        launch_result.project_name = job_config.project_name
        launch_result.application_name = job_config.application_name


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


def _parse_schedule_result(result, yaml_file):
    if result is None:
        return (ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED],
                ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED)
    if result.job_url == "":
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_JOB_URL_ERROR],
                ApiConstants.RESOURCE_MATCHED_STATUS_JOB_URL_ERROR)
    if result.status == Constants.JOB_START_STATUS_LAUNCHED:
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED],
                ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED)
    if result.status == Constants.JOB_START_STATUS_INVALID:
        return (ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_INVALID],
                f"\nPlease check your {os.path.basename(yaml_file)} file "
                f"to make sure the syntax is valid, e.g. "
                f"whether minimum_num_gpus or maximum_cost_per_hour is valid.")
    elif result.status == Constants.JOB_START_STATUS_BLOCKED:
        return (ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_BLOCKED],
                f"\nBecause the value of maximum_cost_per_hour is too low, we can not find exactly matched machines "
                f"for your job. \n")
    elif result.status == Constants.JOB_START_STATUS_QUEUED:
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_QUEUED],
                f"\nNo resource available now, job queued in waiting queue.")
    elif result.status == Constants.JOB_START_STATUS_BIND_CREDIT_CARD_FIRST:
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_BIND_CREDIT_CARD_FIRST],
                ApiConstants.RESOURCE_MATCHED_STATUS_BIND_CREDIT_CARD_FIRST)
    elif result.status == Constants.JOB_START_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED:
        return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED],
                ApiConstants.RESOURCE_MATCHED_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED)

    return (ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED],
            ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED)
