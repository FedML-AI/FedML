import collections
import click

import fedml
from fedml.cli.modules import login, logs, launch, diagnosis, logout, build, run, model, device, cluster, \
    run, train, federate, storage
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

# Add device subcommand module
cli.add_command(device.fedml_device)

# Add model subcommand module
cli.add_command(model.fedml_model)

# Add build subcommand module
cli.add_command(build.fedml_build)

# Add logs subcommand module
cli.add_command(logs.fedml_logs)

# Add train subcommand module
cli.add_command(train.fedml_train)

# Add federate subcommand module
cli.add_command(federate.fedml_federate)

# Add dataset subcommand module
cli.add_command(storage.fedml_storage)


@cli.command(
    "env",
    help="Get environment info such as versions, hardware, and networking",
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


@cli.command("version", help="Display FEDML library version")
@click.help_option("--help", "-h")
def fedml_version():
    click.echo("fedml version: " + str(fedml.__version__))


if __name__ == "__main__":
    cli()
