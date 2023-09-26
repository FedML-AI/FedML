import click

import fedml.api
from fedml.cli.modules.utils import OrderedGroup


@click.group("model", cls=OrderedGroup)
@click.help_option("--help", "-h")
def fedml_model():
    """
    Deploy and infer models.
    """
    pass


@fedml_model.command("create", help="Create local model repository")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--config_file", "-cf", default = None,type=str, help="Model config file (.yaml)",
)
def fedml_model_create(name, config_file):
    fedml.api.model_create(name, config_file)


@fedml_model.command("push", help="Push local model repository to the MLOps platform (cloud)")
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


@fedml_model.command("deploy", help="Deploy model to the local machine or MLOps platform (cloud)")
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
    "run", help="Run inference action for model from the MLOps platform")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--data", "-d", type=str, help="input data for model inference.",
)
def fedml_model_run(name, data):
    fedml.api.model_run(name, data)



@fedml_model.command("pull", help="Pull remote model to local model repository")
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



@fedml_model.command("list", help="List model in the local model repository")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_list(name):
    fedml.api.model_list(name)


@fedml_model.command("list-remote", help="List models in the remote model repository")
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



@fedml_model.command(
    "info", help="Get model information from MLOps platform")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_info(name):
    fedml.api.model_info(name)



@fedml_model.command("delete", help="Delete local model repository")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_delete(name):
    fedml.api.model_delete(name)
