import collections
import click

import fedml
from fedml.cli.modules import login, logs, launch, diagnosis, logout, build, job, model, device, cluster, run
from fedml.cli.modules.utils import OrderedGroup
from fedml.computing.scheduler.env.collect_env import collect_env


@click.group(cls=OrderedGroup)
@click.help_option("--help", "-h")
def cli():
    pass


# Add login subcommand module
cli.add_command(login.fedml_login)


# Add logout subcommand module
cli.add_command(logout.fedml_logout)


# Add launch subcommand module
cli.add_command(launch.fedml_launch)

# Add cluster subcommand module
cli.add_command(cluster.fedml_clusters)

# Add run subcommand module
cli.add_command(run.fedml_run)

# Add job subcommand module
cli.add_command(job.fedml_jobs)


# Add device subcommand module
cli.add_command(device.fedml_device)


# Add model subcommand module
cli.add_command(model.fedml_model)


# Add build subcommand module
cli.add_command(build.fedml_build)

# Add logs subcommand module
cli.add_command(logs.fedml_logs)


@cli.command(
    "env",
    help="collect the environment information to help debugging, including OS, Hardware Architecture, "
         "Python version, etc.",
)
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="support values: local, dev, test, release",
)
def fedml_env(version):
    fedml.set_env_version(version)
    collect_env()


# Add diagnosis subcommand module
cli.add_command(diagnosis.fedml_diagnosis)


@cli.command("version", help="Display fedml version.")
@click.help_option("--help", "-h")
def fedml_version():
    click.echo("fedml version: " + str(fedml.__version__))


if __name__ == "__main__":
    cli()
