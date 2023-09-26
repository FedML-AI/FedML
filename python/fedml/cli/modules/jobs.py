import click

import fedml.api

from fedml.computing.scheduler.slave.client_constants import ClientConstants


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
    fedml.api.stop_job(job_id=job_id, version=version, platform=platform, api_key=api_key)



@fedml_jobs.command("status", help="Display fedml client training status.")
@click.help_option("--help", "-h")
def fedml_status():
    training_infos = ClientConstants.get_training_infos()
    click.echo(
        "Job status: " + str(training_infos["training_status"]).upper()
    )



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
    fedml.api.list_jobs(api_key, version, job_name, job_id, platform)
