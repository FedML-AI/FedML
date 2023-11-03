import click

import fedml.api

from prettytable import PrettyTable


@click.group("run")
@click.help_option("--help", "-h")
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
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the FedML® Nexus AI Platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
def fedml_run(api_key, version, platform):
    """
    Manage runs on the FedML® Nexus AI Platform.
    """
    pass


@fedml_run.command("stop", help="Stop a run from the FedML® Nexus AI Platform.")
@click.help_option("--help", "-h")
@click.option(
    "--run_id",
    "-rid",
    type=str,
    default="",
    help="Id of the run.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="stop a run at which version of FedML® Nexus AI Platform. It should be dev, test or release",
)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the FedML® Nexus AI Platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
def stop_run(platform, run_id, api_key, version):
    fedml.set_env_version(version)
    is_stopped = fedml.api.run_stop(run_id=run_id, platform=platform, api_key=api_key)
    if is_stopped:
        click.echo(f"Run {run_id} is stopped successfully.")
    else:
        click.echo(f"Failed to stop Run {run_id}. Please check if the run id is valid.")


@fedml_run.command("list", help="List runs from the FedML® Nexus AI Platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the FedML® Nexus AI Platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--run_name",
    "-r",
    type=str,
    default="",
    help="Run name at the FedML® Nexus AI Platform.",
)
@click.option(
    "--run_id",
    "-rid",
    type=str,
    default="",
    help="Run id at the FedML® Nexus AI Platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="list runs at which version of FedML® Nexus AI Platform. It should be dev, test or release",
)
def list_runs(platform, run_name, run_id, api_key, version):
    fedml.set_env_version(version)
    run_list_obj = fedml.api.run_list(api_key=api_key, run_name=run_name, run_id=run_id,
                                      platform=platform)
    _print_run_table(run_list_obj)


@fedml_run.command("status", help="Get status of run from the FedML® Nexus AI Platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the FedML® Nexus AI Platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--run_name",
    "-r",
    type=str,
    default=None,
    help="Run name at the FedML® Nexus AI Platform.",
)
@click.option(
    "--run_id",
    "-rid",
    type=str,
    default=None,
    help="Run id at the FedML® Nexus AI Platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="get status of run at which version of FedML® Nexus AI Platform. It should be dev, test or release",
)
def status(platform, run_name, run_id, api_key, version):
    fedml.set_env_version(version)
    if run_name is None and run_id is None:
        click.echo("Please specify run name or run id.")
        return
    run_list_obj, _ = fedml.api.run_status(api_key=api_key, run_name=run_name, run_id=run_id,
                                           platform=platform)
    _print_run_table(run_list_obj)


@fedml_run.command("logs", help="Get logs of run from the FedML® Nexus AI Platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the FedML® Nexus AI Platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--run_id",
    "-rid",
    type=str,
    default=None,
    help="Run id at the FedML® Nexus AI Platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="get logs of run at which version of FedML® Nexus AI Platform. It should be dev, test or release",
)
@click.option(
    "--page_num",
    "-pn",
    type=int,
    default=0,
    help="request page num for logs. --need_all_logs should be set to False if you want to use this option.",
)
@click.option(
    "--page_size",
    "-ps",
    type=int,
    default=0,
    help="request page size for logs, --need_all_logs should be set to False if you want to use this option.",
)
@click.option(
    "--need_all_logs",
    "-a",
    type=bool,
    default=True,
    help="boolean value representing if all logs are needed. Default to True",
)
def logs(platform, run_id, api_key, version, page_num, page_size, need_all_logs):
    fedml.set_env_version(version)
    if run_id is None:
        click.echo("Please specify run id.")
        return

    run_log_result = fedml.api.run_logs(run_id=run_id, page_num=page_num, page_size=page_size,
                                        need_all_logs=need_all_logs, platform=platform, api_key=api_key)

    run_logs = run_log_result.run_logs
    if run_log_result.run_logs is None:
        click.echo(f"No logs found for Run id: {run_id}. "
                   f"Please double check your arguments and make sure they are valid. -h for help.")
        return

    # Show run log summary info
    log_head_table = PrettyTable(['Run ID', 'Status', 'Total Log Lines', 'Log URL'])
    log_head_table.add_row([run_id, run_log_result.run_status, run_log_result.total_log_lines, run_logs.log_full_url])
    click.echo("\nLogs summary info is as follows.")
    print(log_head_table)

    # Show run logs URL for each device
    if len(run_logs.log_devices) > 0:
        log_device_table = PrettyTable(['Device ID', 'Device Name', 'Device Log URL'])
        for log_device in run_logs.log_devices:
            log_device_table.add_row([log_device.device_id, log_device.device_name, log_device.log_url])
        click.echo("\nLogs URL for each device is as follows.")
        print(log_device_table)

    # Show run log lines
    if len(run_log_result.log_line_list) > 0:
        click.echo("\nAll logs is as follows.")
        for log_line in run_log_result.log_line_list:
            click.echo(log_line.rstrip('\n'))


def _print_run_table(run_list_obj):
    if run_list_obj is not None and len(run_list_obj.run_list) > 0:

        click.echo("Found the following matched runs.")
        run_list_table = PrettyTable(['Run Name', 'Run ID', 'Status',
                                      'Created', 'Spend Time(hour)', 'Cost'])
        runs_count = 0
        for run in run_list_obj.run_list:
            runs_count += 1
            run_list_table.add_row([run.run_name, run.run_id, run.status, run.started_time,
                                    run.compute_duration, run.cost])

        click.echo(run_list_table)
    else:
        click.echo("No runs found")
