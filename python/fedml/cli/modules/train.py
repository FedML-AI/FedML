import click

import fedml.api


@click.group("train")
@click.help_option("--help", "-h")
def fedml_train():
    """
    Manage training resources on FedML® Nexus AI Platform
    """
    pass


@fedml_train.command("build", help="Build training packages for the FedML® Nexus AI Platform.")
@click.help_option("--help", "-h")
@click.option(
    "--dest_folder",
    "-d",
    type=str,
    default=None,
    help="The destination package folder path. "
         "If this option is not specified, the built packages will be located "
         "in a subdirectory named fedml-train-packages in the directory of YAML_FILE",
)
@click.argument("yaml_file", nargs=-1)
def build(yaml_file, dest_folder):
    fedml.api.train_build(yaml_file[0], dest_folder)
