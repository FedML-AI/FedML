import click

import fedml.api


@click.group("model")
@click.help_option("--help", "-h")
def fedml_model():
    """
    Deploy and infer models.
    """
    pass


@fedml_model.command("create", help="Create local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--config_file", "-cf", default = None,type=str, help="Model config file (.yaml)",
)
def fedml_model_create(name, config_file):
    fedml.api.model_create(name, config_file)

@fedml_model.command("delete", help="Delete local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_delete(name):
    fedml.api.model_delete(name)


@fedml_model.command("add", help="Add file to local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--path", "-p", type=str, help="path for specific model.",
)
def fedml_model_add_files(name, path):
    fedml.api.model_add_files(name, path)


@fedml_model.command("remove", help="Remove file from local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--file", "-f", type=str, help="file name for specific model.",
)
def fedml_model_remove_files(name, file):
    fedml.api.model_remove_files(name, file)


@fedml_model.command("list", help="List model in the local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_list(name):
    fedml.api.model_list(name)


@fedml_model.command("list-remote", help="List models in the remote model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
def fedml_model_list_remote(name, user, api_key, version, local_server):
    fedml.api.model_list_remote(name, user, api_key, version, local_server)


@fedml_model.command("package", help="Build local model repository as zip model package.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_package(name):
    fedml.api.model_package(name)


@fedml_model.command("push", help="Push local model repository to ModelOps(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--model_storage_url", "-s", type=str, help="model storage url.",
)
@click.option(
    "--model_net_url", "-mn", type=str, help="model net url.",
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
def fedml_model_push(name, model_storage_url, model_net_url, user, api_key, version, local_server):
    fedml.api.model_push(name, model_storage_url, model_net_url, user, api_key, version, local_server)


@fedml_model.command("pull", help="Pull remote model(ModelOps) to local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
def fedml_model_pull(name, user, api_key, version, local_server):
    fedml.api.model_pull(name, user, api_key, version, local_server)


@fedml_model.command("deploy", help="Deploy model to local machine or ModelOps platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="[Required] Model Cards Name.", required=True
)
@click.option(
    "--local", "-l", default=False, is_flag=True, help="Deploy model locally.",
)
@click.option(
    "--master_ids", "-m", type=str, default="", help="[Optional] For on-premise deploy mode, Please indicate master device id(s), seperated with ','"
)
@click.option(
    "--worker_ids", "-w", type=str, default="", help="[Optional] For on-premise deploy mode, Please indicate worker device id(s), seperated with ','"
)
@click.option(
    "--user_id", "-u", type=str, default="", help="[Optional] For on-premise deploy mode, Please indicate user id"
)
@click.option(
    "--api_key", "-k", type=str, default="", help="[Optional] For on-premise deploy mode, Please indicate api key"
)
def fedml_model_deploy(local, name, master_ids, worker_ids, user_id, api_key):
    fedml.api.model_deploy(local, name, master_ids, worker_ids, user_id, api_key)


@fedml_model.command(
    "info", help="Get information of specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_info(name):
    fedml.api.model_info(name)


@fedml_model.command(
    "run", help="Run inference action for specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--data", "-d", type=str, help="input data for model inference.",
)
def fedml_model_run(name, data):
    fedml.api.model_run(name, data)


@fedml_model.command("show-resource-type", help="Show resource type at the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="show resource type at which version of FedML® Launch platform. It should be dev, test or release",
)
def resource_type(version):
    fedml.api.resource_type(version)