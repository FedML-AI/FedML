import click
import os

from prettytable import PrettyTable

import fedml
from fedml.cli.modules.utils import DefaultCommandGroup
from fedml.api.constants import ApiConstants
from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml import set_env_version
from fedml.api.modules.launch import (create_run, create_run_on_cluster, run)
from fedml.api.modules.cluster import confirm_and_start
from fedml.api import run_stop, run_list
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunStartedModel, FeatureEntryPoint


class LaunchGroup(DefaultCommandGroup):
    def format_usage(self, ctx, formatter):
        click.echo("fedml launch [OPTIONS] YAML_FILE")


@click.group("launch", cls=LaunchGroup, default_command='default')
@click.help_option("--help", "-h")
@click.option(
    "--cluster",
    "-c",
    default="",
    type=str,
    help="If a cluster name is specified, you labelled the searched resource by launch with the cluster name. So later you can reuse the same cluster resource without warmup after the first launch. The cluster can be stopped by CLI: fedml cluster stop, or it would be automatically stopped after 15-minute idle time."
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="version of FedML® Nexus AI Platform. It should be dev, test or release",
)
def fedml_launch(api_key, version, cluster):
    """
    Launch job at the FedML® Nexus AI platform
    """
    pass


@fedml_launch.command(
    "default", help="Launch job at the FedML® Nexus AI Platform",
    context_settings={"ignore_unknown_options": True}, hidden=True
)
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--group",
    "-g",
    type=str,
    default="",
    help="The queue group id on which your job will be scheduled.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="launch job to which version of MLOps platform. It should be dev, test or release",
)
@click.option(
    "--cluster",
    "-c",
    default=None,
    type=str,
    help="If a cluster name is specified, you labelled the searched resource by launch with the cluster name. So later you can reuse the same cluster resource without warmup after the first launch. The cluster can be stopped by CLI: fedml cluster stop, or it would be automatically stopped after 15-minute idle time."
)
@click.argument("yaml_file", nargs=-1)
def fedml_launch_default(yaml_file, api_key, group, cluster, version):
    """
    Manage resources on the FedML® Nexus AI Platform.
    """
    set_env_version(version)

    if cluster is None:
        _launch_job(yaml_file[0], api_key)
    else:
        _launch_job_on_cluster(yaml_file[0], api_key, cluster)


def _launch_job(yaml_file, api_key):
    result_code, result_message, create_run_result = create_run(
        yaml_file, api_key=api_key, feature_entry_point=FeatureEntryPoint.FEATURE_ENTRYPOINT_CLI)

    if _resources_matched_and_confirmed(result_code, result_message, create_run_result, yaml_file, api_key):
        if create_run_result.user_check:
            if not click.confirm("Do you want to launch the job with the above matched GPU "
                                 "resource?", abort=False):
                click.echo("Cancelling the job with the above matched GPU resource.")
                run_stop(create_run_result.run_id, SchedulerConstants.PLATFORM_TYPE_FALCON, api_key=api_key)
                return False

        click.echo("Launching the job with the above matched GPU resource.")
        result = run(create_run_result=create_run_result, api_key=api_key)
        _print_run_list_details(result)
        _print_run_log_details(result)


def _launch_job_on_cluster(yaml_file, api_key, cluster):
    result_code, result_message, create_run_result = create_run_on_cluster(
        yaml_file, cluster, api_key, feature_entry_point=FeatureEntryPoint.FEATURE_ENTRYPOINT_CLI)
    cluster_confirmed = True
    if _resources_matched_and_confirmed(result_code=result_code, result_message=result_message,
                                        create_run_result=create_run_result, yaml_file=yaml_file, api_key=api_key):
        if create_run_result.user_check:
            if not click.confirm("Do you want to launch the job with the above matched GPU "
                                 "resource?", abort=False):
                click.echo("Cancelling the job with the above matched GPU resource.")
                run_stop(run_id=create_run_result.run_id, platform=SchedulerConstants.PLATFORM_TYPE_FALCON,
                         api_key=api_key)
                return False

            cluster_id = getattr(create_run_result, "cluster_id", None)

            if cluster_id is None or cluster_id == "":
                click.echo("Cluster id was not assigned. Please check if the cli arguments are valid")
                return

            cluster_confirmed = confirm_and_start(run_id=create_run_result.run_id, cluster_id=cluster_id,
                                                  gpu_matched=create_run_result.gpu_matched)

        if cluster_confirmed:
            _print_run_list_details(create_run_result)
            _print_run_log_details(create_run_result)


def _check_match_result(result: FedMLRunStartedModel, yaml_file: dict):
    if result.run_url == "":
        if result.message is not None:
            click.echo(f"Failed to launch the job with response messages: {result.message}")
        else:
            click.echo(f"Failed to launch the job. Please check if the network is available.")

        return ApiConstants.RESOURCE_MATCHED_STATUS_JOB_URL_ERROR

    if result.status == Constants.JOB_START_STATUS_LAUNCHED:
        return ApiConstants.LAUNCH_JOB_STATUS_REQUEST_SUCCESS
    if result.status == Constants.JOB_START_STATUS_INVALID:
        click.echo(f"\nPlease check your {os.path.basename(yaml_file)} file "
                   f"to make sure the syntax is valid, e.g. "
                   f"whether minimum_num_gpus or maximum_cost_per_hour is valid.")
        return ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED
    elif result.status == Constants.JOB_START_STATUS_BLOCKED:
        click.echo("\nBecause the value of maximum_cost_per_hour is too low,"
                   "we can not find exactly matched machines for your job. \n"
                   "But here we still present machines closest to your expected price as below.")
        return ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED
    elif result.status == Constants.JOB_START_STATUS_QUEUED:
        click.echo("\nNo resource available now, but we can keep your job in the waiting queue.")
        if click.confirm("Do you want to join the queue?", abort=False):
            click.echo("You have confirmed to keep your job in the waiting list.")
            _print_run_list_details(result)
            return ApiConstants.RESOURCE_MATCHED_STATUS_QUEUED
        else:
            click.echo("Cancelling launch as no resources are available. Please try again later.")
            return ApiConstants.RESOURCE_MATCHED_STATUS_QUEUE_CANCELED
    elif result.status == Constants.JOB_START_STATUS_BIND_CREDIT_CARD_FIRST:
        click.echo("Please bind your credit card before launching the job.")
        return ApiConstants.RESOURCE_MATCHED_STATUS_BIND_CREDIT_CARD_FIRST
    elif result.status == Constants.JOB_START_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED:
        click.echo("Failed to query credit card binding status. Please try again later.")
        return ApiConstants.RESOURCE_MATCHED_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED

    return ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED


def _match_and_show_resources(result: FedMLRunStartedModel):
    gpu_matched = getattr(result, "gpu_matched", None)
    if gpu_matched is not None and len(gpu_matched) > 0:
        click.echo(f"\nSearched and matched the following GPU resource for your job:")
        gpu_table = PrettyTable(['Provider', 'Instance', 'vCPU(s)', 'Memory(GB)', 'GPU(s)',
                                 'Region', 'Cost', 'Selected'])
        for gpu_device in gpu_matched:
            gpu_table.add_row([gpu_device.gpu_provider, gpu_device.gpu_instance, gpu_device.cpu_count,
                               gpu_device.mem_size,
                               f"{gpu_device.gpu_type}:{gpu_device.gpu_num}",
                               gpu_device.gpu_region, gpu_device.cost, Constants.CHECK_MARK_STRING])
        click.echo(gpu_table)
        click.echo("")

        click.echo(f"You can also view the matched GPU resource with Web UI at: ")
        click.echo(f"{result.run_url}")


def _resources_matched_and_confirmed(result_code: int, result_message: str, create_run_result: FedMLRunStartedModel,
                                     yaml_file: dict, api_key: str):
    if result_code == ApiConstants.ERROR_CODE[ApiConstants.APP_UPDATE_FAILED] or not create_run_result:
        click.echo(f"{result_message}. Please double check the input arguments are valid.")
        return False
    match_result = _check_match_result(create_run_result, yaml_file)
    if match_result == ApiConstants.RESOURCE_MATCHED_STATUS_QUEUE_CANCELED:
        run_stop(run_id=create_run_result.run_id, platform=SchedulerConstants.PLATFORM_TYPE_FALCON, api_key=api_key)
        return False
    if (match_result == ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED or
            match_result == ApiConstants.LAUNCH_JOB_STATUS_REQUEST_SUCCESS):
        _match_and_show_resources(create_run_result)
        return True
    return False


def _print_run_list_details(result):
    if result is None:
        click.echo("Failed to launch the job")
        return

    if result.run_url == "":
        if result.message is not None:
            click.echo(f"Failed to launch the job with response messages: {result.message}")
        else:
            click.echo("Failed to launch the job")

    # List the run status
    run_list_obj = run_list(run_name=result.project_name, platform=SchedulerConstants.PLATFORM_TYPE_FALCON,
                            run_id=result.run_id)
    if run_list_obj is not None and len(run_list_obj.run_list) > 0:
        click.echo("")
        click.echo("Your run result is as follows:")
        run_list_table = PrettyTable(['Run Name', 'Run ID', 'Status', 'Created',
                                      'Spend Time(hour)', 'Cost'])
        runs_count = 0
        for run in run_list_obj.run_list:
            runs_count += 1
            run_list_table.add_row([run.run_name, run.run_id, run.status, run.started_time,
                                    run.compute_duration, run.cost])
        click.echo(run_list_table)
    else:
        click.echo("")

    # Show the run url
    click.echo("\nYou can track your run details at this URL:")
    click.echo(f"{result.run_url}")


def _print_run_log_details(result: FedMLRunStartedModel):
    # Show the run url
    if not (result and result.run_id):
        return

    # Show querying infos for getting run logs
    click.echo("")
    click.echo(f"For querying the realtime status of your run, please run the following command.")
    click.echo(f"fedml run logs -rid {result.run_id}" +
               "{}".format(f" -v {fedml.get_env_version()}"))

