import click
from prettytable import PrettyTable

import fedml
from fedml.cli.modules import login, logs, launch, diagnosis, logout, build, jobs, model, device, inference
from fedml.computing.scheduler.env.collect_env import collect_env
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager
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


@cli.command("show-resource-type", help="Show resource type at the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="show resource type at which version of FedML® Launch platform. It should be dev, test or release",
)
def fedml_show_resource_type(version):
    FedMLLaunchManager.get_instance().set_config_version(version)
    resource_type_list = FedMLLaunchManager.get_instance().show_resource_type()
    if resource_type_list is not None and len(resource_type_list) > 0:
        click.echo("All available resource type is as follows.")
        resource_table = PrettyTable(['Resource Type', 'GPU Type'])
        for type_item in resource_type_list:
            resource_table.add_row([type_item[0], type_item[1]])
        print(resource_table)
    else:
        click.echo("No available resource type.")


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

# Add inference subcommand module
cli.add_command(inference.fedml_model_inference)

# Add launch subcommand module
cli.add_command(launch.fedml_launch)

if __name__ == "__main__":
    cli()