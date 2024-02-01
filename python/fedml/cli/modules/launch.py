import click

from prettytable import PrettyTable

import fedml
from fedml.api.constants import ApiConstants
from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml import set_env_version
from fedml.api.modules.launch import (create_run, create_run_on_cluster, run)
from fedml.api.modules.cluster import confirm_and_start
from fedml.api import run_stop, run_list
from fedml.computing.scheduler.scheduler_entry.run_manager import FedMLRunStartedModel, FeatureEntryPoint


@click.command("launch", help="Launch job at the FedML® Nexus AI Platform")
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
@click.option(
    "--local_on_premise_platform",
    "-lp",
    type=str,
    default="127.0.0.1",
    help="The IP address for local on-premise Nexus AI Platform.",
)
@click.option(
    "--local_on_premise_platform_port",
    "-lpp",
    type=int,
    default=80,
    help="The port for local on-premise Nexus AI Platform.",
)
@click.argument("yaml_file", nargs=-1)
def fedml_launch(yaml_file, cluster, version, api_key, group, local_on_premise_platform, local_on_premise_platform_port):
    """
    Manage resources on the FedML® Nexus AI Platform.
    """
    set_env_version(version)
    fedml.set_local_on_premise_platform_host(local_on_premise_platform)
    fedml.set_local_on_premise_platform_port(local_on_premise_platform_port)

    # Input YAML file is recognized as tuple ().
    # If empty, print help msg and exit.
    if not yaml_file:        
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()        
    else:
        if cluster is None:
            _launch_job(yaml_file[0], api_key)
        else:
            _launch_job_on_cluster(yaml_file[0], api_key, cluster)


def _launch_job(yaml_file, api_key):
    result_code, result_message, create_run_result = create_run(
        yaml_file, api_key=api_key, feature_entry_point=FeatureEntryPoint.FEATURE_ENTRYPOINT_CLI)

    if _resources_matched(result_code=result_code, result_message=result_message,
                          create_run_result=create_run_result, api_key=api_key):

        if create_run_result.user_check:
            if not click.confirm("Do you want to launch the job with the above matched GPU "
                                 "resource?", abort=False):
                click.echo("Cancelling the job with the above matched GPU resource.")
                run_stop(create_run_result.run_id, SchedulerConstants.PLATFORM_TYPE_FALCON, api_key=api_key)
                return False

        # Print Details
        click.echo("Launching the job with the above matched GPU resource.")
        result = run(create_run_result=create_run_result, api_key=api_key)
        _print_run_list_details(result)


def _launch_job_on_cluster(yaml_file, api_key, cluster):
    result_code, result_message, create_run_result = create_run_on_cluster(
        yaml_file, cluster, api_key, feature_entry_point=FeatureEntryPoint.FEATURE_ENTRYPOINT_CLI)

    if _resources_matched(result_code=result_code, result_message=result_message,
                          create_run_result=create_run_result, api_key=api_key):

        if create_run_result.user_check:

            # Confirm Cluster and Start
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

            confirm_and_start(run_id=create_run_result.run_id, cluster_id=cluster_id,
                              gpu_matched=create_run_result.gpu_matched)

        # Print Details
        click.echo("Launching the job with the above matched GPU resource.")
        _print_run_list_details(create_run_result)


def _resources_matched(result_code: int, result_message: str, create_run_result: FedMLRunStartedModel,
                       api_key: str):
    if result_code == ApiConstants.ERROR_CODE[ApiConstants.APP_UPDATE_FAILED] or not create_run_result:
        click.echo(f"Result Message: {result_message}. Please double check the input arguments are valid.")
        return False

    if result_code == ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_QUEUED]:
        if create_run_result.job_type == SchedulerConstants.JOB_TASK_TYPE_DEPLOY or \
                create_run_result.job_type == SchedulerConstants.JOB_TASK_TYPE_SERVE:
            click.echo(f"\nNo resource available now, please modify the resource type or try it again later.")
            return False
        click.echo("\nNo resource available now, but we can keep your job in the waiting queue and run your job "
                   "whenever requested resources are available.")
        if click.confirm("Do you want to join the queue?", abort=False):
            click.echo("You have confirmed to keep your job in the waiting list.")
            _print_run_list_details(create_run_result)
            return False
        else:
            click.echo("Cancelling launch as no resources are available. Please try again later.")
            run_stop(run_id=create_run_result.run_id, platform=SchedulerConstants.PLATFORM_TYPE_FALCON, api_key=api_key)
            return False

    if result_code == ApiConstants.ERROR_CODE[ApiConstants.LAUNCHED] or \
            result_code == ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED]:
        _match_and_show_resources(create_run_result)
        return True

    click.echo(result_message)
    return False


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


def _print_run_list_details(result: FedMLRunStartedModel):
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

    click.echo("")

    # Show the run url
    click.echo(f"You can query the realtime status and logs of your with the help of following tools / commands:")
    run_track_table = PrettyTable(['Description', 'Details'])
    run_track_table.add_row(["Run Track URL (UI):", '{}'.format(result.run_url)])

    if result.run_id:
        version = fedml.get_env_version()
        run_track_table.add_row(["Run Track Command (CLI):", f"fedml run logs -rid {result.run_id}" +
                                 "{}".format(f" -v {version}" if version != "release" else "")])

    click.echo(run_track_table)
