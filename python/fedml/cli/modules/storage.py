import ast
import urllib
from urllib.parse import unquote
from prettytable import PrettyTable

import click

import fedml.api
import pprint

from fedml.api import StorageMetadata
from fedml.api.fedml_response import ResponseCode

# Message strings constants
version_help: str = "specify version of FedML® Nexus AI Platform. It should be dev, test or release"
api_key_help: str = "user api key."


# Todo (alaydshah): Add support to update already stored data
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
@click.option("--description", "-d", type=str, help="Add description to your data to store. If not provided, "
                                                    "the description will be empty.")
@click.option("--user_metadata", "-um", type=str, help="User-defined metadata in the form of a dictionary, for instance, "
                                                       " {'name':'value'} within double quotes. "" "
                                                       "Defaults to None.")
@click.option("--tags", "-t", type=str, help="Add tags to your data to store. Give tags in comma separated form like 'cv,unet,segmentation' If not provided, the tags will be empty.")
@click.option('--service', "-s", type=click.Choice(['R2']), default="R2", help="Storage service for object storage. "
                                                                               "Only R2 is supported as of now")
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
def upload(data_path: str, name: str, user_metadata: str, description: str, version: str, api_key: str, tags:str, service):
    metadata = _parse_metadata(user_metadata)
    tag_list = _parse_tags(tags)
    fedml.set_env_version(version)
    response = fedml.api.upload(data_path=data_path, api_key=api_key, name=name, tag_list = tag_list, service=service, show_progress=True,
                                description=description, metadata=metadata)
    if response.code == ResponseCode.SUCCESS:
        click.echo(f"Data uploaded successfully. | url: {response.data}")
    else:
        click.echo(f"Failed to upload data. Error message: {response.message}")


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
    response = fedml.api.list_storage_objects(api_key=api_key)
    if response.code == ResponseCode.SUCCESS:
        click.echo(f"Successfully fetched list of stored objects:")
        if not response.data:
            click.echo(f"No stored objects found for account linked with apikey: {api_key}")
            return
        object_list_table = PrettyTable(["Data Name", "Data Size", "Description", "Data Tags","Created At", "Updated At"])
        for stored_object in response.data:
            object_list_table.add_row(
                [stored_object.dataName, stored_object.size, stored_object.description, stored_object.tag_list,stored_object.createdAt, stored_object.updatedAt])
        click.echo(object_list_table)
    else:
        click.echo(f"Failed to list stored objects for account linked with apikey {api_key}. "
                   f"Error message: {response.message}")


@fedml_storage.command("get-user-metadata", help="Get user-defined metadata of data object stored on FedML® Nexus AI "
                                                 "Platform")
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
def get_user_metadata(data_name, version, api_key):
    fedml.set_env_version(version)
    response = fedml.api.get_storage_user_defined_metadata(data_name=data_name, api_key=api_key)
    if response.code == ResponseCode.SUCCESS:
        if not response.data:
            click.echo(f"No user-metadata exists for {data_name}")
            return
        click.echo(f"Successfully fetched user-metadata for {data_name}:")
        pprint.pprint(response.data)

    else:
        click.echo(f"Failed to fetch user-metadata for {data_name}. Error message: {response.message}")


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
    response = fedml.api.get_storage_metadata(api_key=api_key, data_name=data_name)
    if response.code == ResponseCode.SUCCESS:
        metadata = response.data
        if not metadata or not isinstance(metadata, StorageMetadata):
            click.echo(f"No metadata exists for object {data_name}")
            return
        click.echo(f"Successfully fetched metadata for object {data_name}:")
        # Todo (alaydshah): Add file size and tags
        metadata_table = PrettyTable(["Data Name","Data Size","Description","Data Tags","Created At", "Updated At"])
        metadata_table.add_row([metadata.dataName,metadata.size,metadata.description,metadata.tag_list,metadata.createdAt, metadata.updatedAt])
        click.echo(metadata_table)
        click.echo("")
    else:
        click.echo(f"Fetching metadata failed. Error message: {response.message}")


@fedml_storage.command("download", help="Download data stored on FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("data_name", nargs=1, callback=validate_argument)
@click.option("--dest_path", "-d", default=None, type=str, help="Destination path to download data. By default, "
                                                                "it would be downloaded to working directory from "
                                                                "where the command is executed.")
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option('--service', "-s", type=click.Choice(['R2']), default="R2", help="Storage service for object storage. "
                                                                               "Only R2 is supported as of now")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def download(data_name, dest_path, version, api_key, service):
    fedml.set_env_version(version)
    response = fedml.api.download(data_name=data_name, dest_path=dest_path, api_key=api_key, service=service)
    if response.code == ResponseCode.SUCCESS:
        click.echo(f"Data downloaded successfully at: {response.data}")
    else:
        click.echo(f"Failed to download data {data_name}. Error message: {response.message}")


@fedml_storage.command("delete", help="Delete data stored on FedML® Nexus AI Platform")
@click.argument("data_name", nargs=1, callback=validate_argument)
@click.help_option("--help", "-h")
@click.option(
    "--api_key", "-k", type=str, help=api_key_help,
)
@click.option('--service', "-s", type=click.Choice(['R2']), default="R2", help="Storage service for object storage. "
                                                                               "Only R2 is supported as of now")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help=version_help,
)
def delete(version, data_name, api_key, service):
    fedml.set_env_version(version)
    response = fedml.api.delete(data_name=data_name, api_key=api_key, service=service)
    if response.code == ResponseCode.SUCCESS:
        click.echo(f"Data '{data_name}' deleted successfully.")
    else:
        click.echo(f"Failed to delete data {data_name}. Error message: {response.message}")


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

def _parse_tags(tags:str):
    if not tags:
        return []
    tag_list = tags.split(",")
    return tag_list 