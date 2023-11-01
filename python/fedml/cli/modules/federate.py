import click

import fedml.api


@click.group("federate")
@click.help_option("--help", "-h")
def fedml_federate():
    """
    Manage federated learning resources on FedML® Nexus AI Platform
    """
    pass


@fedml_federate.command("build", help="Build federate packages for the FedML® Nexus AI Platform.")
@click.help_option("--help", "-h")
@click.option(
    "--dest_folder",
    "-d",
    type=str,
    default=None,
    help="The destination package folder path. "
         "If this option is not specified, the built packages will be located "
         "in a subdirectory named fedml-federate-packages in the directory of YAML_FILE",
)
@click.argument("yaml_file", nargs=-1)
def build(yaml_file, dest_folder):
    fedml.api.federate_build(yaml_file[0], dest_folder)