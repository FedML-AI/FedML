import click

import fedml.api

from prettytable import PrettyTable


@click.group("job")
@click.help_option("--help", "-h")
def fedml_jobs():
    """
    Manage jobs on the MLOps platform.
    """
    pass


@fedml_jobs.command("stop", help="Stop a job from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--job_id",
    "-jid",
    type=str,
    default="",
    help="Id of the job.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="stop a job at which version of MLOps platform. It should be dev, test or release",
)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
def stop_job(platform, job_id, api_key, version):
    fedml.set_env_version(version)
    fedml.api.job_stop(job_id=job_id, platform=platform, api_key=api_key)


@fedml_jobs.command("list", help="List jobs from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--job_name",
    "-j",
    type=str,
    default="",
    help="Job name at the MLOps platform.",
)
@click.option(
    "--job_id",
    "-jid",
    type=str,
    default="",
    help="Job id at the MLOps platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="list jobs at which version of MLOps platform. It should be dev, test or release",
)
def list_jobs(platform, job_name, job_id, api_key, version):
    fedml.set_env_version(version)
    job_list_obj = fedml.api.job_list(api_key=api_key, job_name=job_name, job_id=job_id,
                                      platform=platform)

    _print_job_table(job_list_obj)


@fedml_jobs.command("status", help="Get status of job from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--job_name",
    "-j",
    type=str,
    default=None,
    help="Job name at the MLOps platform.",
)
@click.option(
    "--job_id",
    "-jid",
    type=str,
    default=None,
    help="Job id at the MLOps platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="get status of job at which version of MLOps platform. It should be dev, test or release",
)
def status(platform, job_name, job_id, api_key, version):
    fedml.set_env_version(version)
    if job_name is None and job_id is None:
        click.echo("Please specify job name or job id.")
        return
    job_list_obj, _ = fedml.api.job_status(api_key=api_key, job_name=job_name, job_id=job_id,
                                           platform=platform)
    _print_job_table(job_list_obj)


@fedml_jobs.command("logs", help="Get logs of job from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--job_id",
    "-jid",
    type=str,
    default=None,
    help="Job id at the MLOps platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="get logs of job at which version of MLOps platform. It should be dev, test or release",
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
    help="boolean value representing if all logs are needed",
)
def logs(platform, job_id, api_key, version, page_num, page_size, need_all_logs):
    fedml.set_env_version(version)
    if job_id is None:
        click.echo("Please specify job id.")
        return

    job_status, total_log_lines, total_log_pages, log_list, job_logs = fedml.api.job_logs(job_id=job_id,
                                                                                          page_num=page_num,
                                                                                          page_size=page_size,
                                                                                          need_all_logs=need_all_logs,
                                                                                          platform=platform,
                                                                                          api_key=api_key)

    if job_logs is None:
        click.echo("Not found any logs. Please double check your arguments and make sure they are valid. -h for help.")
        return

    # Show job log summary info
    log_head_table = PrettyTable(['Job ID', 'Total Log Lines', 'Log URL'])
    log_head_table.add_row([job_id, total_log_lines, job_logs.log_full_url])
    click.echo("\nLogs summary info is as follows.")
    print(log_head_table)

    # Show job logs URL for each device
    if len(job_logs.log_devices) > 0:
        log_device_table = PrettyTable(['Device ID', 'Device Name', 'Device Log URL'])
        for log_device in job_logs.log_devices:
            log_device_table.add_row([log_device.device_id, log_device.device_name, log_device.log_url])
        click.echo("\nLogs URL for each device is as follows.")
        print(log_device_table)

    # Show job log lines
    if len(log_list) > 0:
        click.echo("\nAll logs is as follows.")
        for log_line in log_list:
            click.echo(log_line.rstrip('\n'))


@fedml_jobs.command("queue", help="View the job queue at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("group_id", nargs=-1)
def fedml_launch_queue(group_id):
    click.echo("this CLI is not implemented yet")
    return


def _print_job_table(job_list_obj):
    if job_list_obj is not None and len(job_list_obj.job_list) > 0:

        click.echo("Found the following matched jobs.")
        job_list_table = PrettyTable(['Job Name', 'Job ID', 'Status',
                                      'Created', 'Spend Time(hour)', 'Cost'])
        jobs_count = 0
        for job in job_list_obj.job_list:
            jobs_count += 1
            job_list_table.add_row([job.job_name, job.job_id, job.status, job.started_time,
                                    job.compute_duration, job.cost])

        click.echo(job_list_table)
    else:
        click.echo("Not found any jobs")
