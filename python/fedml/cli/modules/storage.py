from typing import List

import click

import fedml.api

# Message strings constants
version_help: str = "specify version of FedML® Nexus AI Platform. It should be dev, test or release"
api_key_help: str = "user api key."


@click.group("storage")
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
def fedml_storage(api_key, version):
    """
    Manage storage on FedML® Nexus AI Platform
    """
    pass


# Callback function to validate argument
def validate_argument(ctx, param, value):
    if not value:
        raise click.BadParameter("No data path provided.")
    return value


@fedml_storage.command("upload", help="Upload data on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("data_path", nargs=1, callback=validate_argument)
@click.option("--name", "-n", type=str, help="Name your data to store. If not provided, the name will be the same as "
                                             "the data file or directory name.")
@click.option("--tags", "-t", type=list, help="Add tags to your data to store. If not provided, the tags "
                                              "will be empty.")
@click.option("--description", "-d", type=str, help="Add description to your data to store. If not provided, "
                                                    "the description will be empty.")

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
def upload(data_path: str, name: str, tags: List[str], description: str, version: str, api_key: str):
    fedml.set_env_version(version)
    storage_url = fedml.api.upload(data_path=data_path, api_key=api_key, name=name, show_progress=True)
    if storage_url:
        click.echo(f"Data uploaded successfully. | url: {storage_url}")
    else:
        click.echo("Failed to upload data.")


@fedml_storage.command("list", help="List data stored on FedML® Nexus AI Platform")
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
def list_data(version, api_key):
    fedml.set_env_version(version)
    click.echo("This feature is actively being worked on. Coming soon...")
    pass


@fedml_storage.command("download", help="Download data stored on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("data_name", nargs=1, callback=validate_argument)
@click.option("--dest_path", "-d", default=None, type=str, help="Destination path to download data.")
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
def download(data_name, dest_path, version, api_key):
    fedml.set_env_version(version)
    data_download_path = fedml.api.download(data_name=data_name, dest_path=dest_path, api_key=api_key)
    if data_download_path:
        click.echo(f"Data downloaded successfully at: {data_download_path}")
    else:
        click.echo(f"Failed to download data {data_name}")
    pass


@fedml_storage.command("delete", help="Delete data stored on FedML® Nexus AI Platform")
@click.argument("data_name", nargs=1, callback=validate_argument)
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
def delete(version, data_name, api_key):
    fedml.set_env_version(version)
    click.echo("This feature is actively being worked on. Coming soon...")
    pass