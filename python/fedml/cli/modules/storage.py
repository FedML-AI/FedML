import ast
import urllib
from typing import List
from urllib.parse import unquote

import click

import fedml.api
import pprint

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
@click.option("--metadata", "-m", type=str, help="Add metadata to your data to store. Defaults to None.")
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
def upload(data_path: str, name: str, metadata: str, tags: List[str], description: str, version: str, api_key: str):
    metadata = _parse_metadata(metadata)
    fedml.set_env_version(version)
    storage_url = fedml.api.upload(data_path=data_path, api_key=api_key, name=name, show_progress=True, metadata=metadata)
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


@fedml_storage.command("get-metadata", help="Get metadata of data object stored on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("data_name", nargs=1, callback=validate_argument)
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
def get_metadata(data_name, version, api_key):
    fedml.set_env_version(version)
    metadata = fedml.api.get_metadata(data_name=data_name, api_key=api_key)
    if not metadata:
        click.echo(f"No metadata exists for object {data_name}")
    else:
        click.echo(f"Successfully fetched metadata for object {data_name}:")
        pprint.pprint(metadata)


@fedml_storage.command("download", help="Download data stored on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("data_name", nargs=1, callback=validate_argument)
@click.option("--dest_path", "-d", default=None, type=str, help="Destination path to download data. By default, "
                                                                "it would be downloaded to working directory from "
                                                                "where the command is executed.")
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


def _parse_metadata(metadata: str):
    if not metadata:
        return {}

    # Unquote the string
    unquoted_string = urllib.parse.unquote(metadata)

    try:
        # Attempt to evaluate the unquoted string as a literal
        result = ast.literal_eval(unquoted_string)

        # Check if the result is a dictionary
        if not isinstance(result, dict):
            click.echo("Error: Metadata should be represented as a dictionary")
            exit()
        return result

    except (SyntaxError, ValueError) as e:
        # Handle the case where the string cannot be evaluated
        click.echo(
            f"Input metadata cannot be evaluated. Please make sure metadata is in the correct format. Error: {e}.")
        exit()

