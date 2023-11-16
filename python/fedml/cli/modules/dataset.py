import click

import fedml.api

# Message strings constants
version_help: str = "specify version of FedML® Nexus AI Platform. It should be dev, test or release"
api_key_help: str = "user api key."


@click.group("dataset")
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def fedml_dataset(api_key, version):
    """
    Manage datasets on FedML® Nexus AI Platform
    """
    pass


# Callback function to validate argument
def validate_argument(ctx, param, value):
    if not value:
        raise click.BadParameter("No dataset provided.")
    return value


@fedml_dataset.command("upload", help="Upload dataset on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("dataset", nargs=1, callback=validate_argument)
@click.option("--name", "-n", type=str, help="Name your dataset. If not provided, the dataset name will be the same as "
                                             "the dataset file name.")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
@click.option(
    "--description",
    "-d",
    type=str,
    default=None,
    help="Description of the dataset",
)
def upload(dataset, version, api_key):
    fedml.set_env_version(version)
    pass


@fedml_dataset.command("list", help="List datasets on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def list_dataset(version, api_key):
    fedml.set_env_version(version)
    pass


@fedml_dataset.command("retrieve-metadata", help="Retrieve metadata of a dataset on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("dataset_id", nargs=1, callback=validate_argument)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def retrieve_metadata(dataset_id, version, api_key):
    fedml.set_env_version(version)
    pass


@fedml_dataset.command("retrieve", help="Retrieve dataset on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("dataset_id", nargs=1, callback=validate_argument)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def retrieve_metadata(dataset_id, version, api_key):
    fedml.set_env_version(version)
    pass


@fedml_dataset.command("delete", help="Delete dataset on FedML® Nexus AI Platform")
@click.argument("dataset_id", nargs=1, callback=validate_argument)
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def delete(version, dataset_id, api_key):
    fedml.set_env_version(version)
    pass
