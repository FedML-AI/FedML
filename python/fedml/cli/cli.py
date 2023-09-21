import click

import fedml
from fedml.cli.modules import login, logs, launch, diagnosis, logout, build, jobs, model, device
from fedml.computing.scheduler.env.collect_env import collect_env
from fedml.computing.scheduler.slave.client_constants import ClientConstants


@click.group()
@click.help_option("--help", "-h")
def cli():
    pass


@cli.command("version", help="Display fedml version.")
@click.help_option("--help", "-h")
def fedml_version():
    click.echo("fedml version: " + str(fedml.__version__))


@cli.command("status", help="Display fedml client training status.")
@click.help_option("--help", "-h")
def fedml_status():
    training_infos = ClientConstants.get_training_infos()
    click.echo(
        "Client training status: " + str(training_infos["training_status"]).upper()
    )


@cli.command(
    "env",
    help="collect the environment information to help debugging, including OS, Hardware Architecture, "
         "Python version, etc.",
)
@click.help_option("--help", "-h")
def fedml_env():
    collect_env()


# Add login subcommand module
cli.add_command(login.fedml_login)

# Add logs subcommand module
cli.add_command(logs.fedml_logs)

# Add diagnosis subcommand module
cli.add_command(diagnosis.fedml_diagnosis)

# Add logout subcommand module
cli.add_command(logout.fedml_logout)

# Add build subcommand module
cli.add_command(build.fedml_build)

# Add job subcommand module
cli.add_command(jobs.fedml_jobs)

# Add device subcommand module
cli.add_command(device.fedml_device)

# Add model subcommand module
cli.add_command(model.fedml_model)

# Add launch subcommand module
cli.add_command(launch.fedml_launch)

if __name__ == "__main__":
    cli()
