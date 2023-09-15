import json
import os

import click

from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards


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
    if config_file is None:
        # Just create a model folder
        if FedMLModelCards.get_instance().create_model(name):
            click.echo("Create model {} successfully.".format(name))
        else:
            click.echo("Failed to create model {}.".format(name))
    else:
        # Adding related workspace codes to the model folder
        if FedMLModelCards.get_instance().create_model_use_config(name, config_file):
            click.echo("Create model {} using config successfully.".format(name))
        else:
            click.echo("Failed to create model {} using config file {}.".format(name, config_file))

@fedml_model.command("delete", help="Delete local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_delete(name):
    if FedMLModelCards.get_instance().delete_model(name):
        click.echo("Delete model {} successfully.".format(name))
    else:
        click.echo("Failed to delete model {}.".format(name))


@fedml_model.command("add", help="Add file to local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--path", "-p", type=str, help="path for specific model.",
)
def fedml_model_add_files(name, path):
    if FedMLModelCards.get_instance().add_model_files(name, path):
        click.echo("Add file to model {} successfully.".format(name))
    else:
        click.echo("Failed to add file to model {}.".format(name))


@fedml_model.command("remove", help="Remove file from local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--file", "-f", type=str, help="file name for specific model.",
)
def fedml_model_remove_files(name, file):
    if FedMLModelCards.get_instance().remove_model_files(name, file):
        click.echo("Remove file from model {} successfully.".format(name))
    else:
        click.echo("Failed to remove file from model {}.".format(name))


@fedml_model.command("list", help="List model in the local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_list(name):
    models = FedMLModelCards.get_instance().list_models(name)
    if len(models) <= 0:
        click.echo("Model list is empty.")
    else:
        for model_item in models:
            click.echo(model_item)
        click.echo("List model {} successfully.".format(name))


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
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    model_query_result = FedMLModelCards.get_instance().list_models(name, user, api_key, local_server)
    if model_query_result is None or model_query_result.model_list is None or len(model_query_result.model_list) <= 0:
        click.echo("Model list is empty.")
    else:
        click.echo("Found {} models:".format(len(model_query_result.model_list)))
        index = 1
        for model_item in model_query_result.model_list:
            model_item.show("{}. ".format(index))
            index += 1
        click.echo("List model {} successfully.".format(name))


@fedml_model.command("package", help="Build local model repository as zip model package.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_package(name):
    model_zip = FedMLModelCards.get_instance().build_model(name)
    if model_zip != "":
        click.echo("Build model package {} successfully".format(name))
        click.echo("The local package is located at {}.".format(model_zip))
    else:
        click.echo("Failed to build model {}.".format(name))


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
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    model_is_from_open = True if model_storage_url is not None and model_storage_url != "" else False
    model_storage_url, model_zip = FedMLModelCards.get_instance().push_model(name, user, api_key,
                                                                             model_storage_url=model_storage_url,
                                                                             model_net_url=model_net_url,
                                                                             local_server=local_server)
    if model_is_from_open:
        click.echo("Push model {} with model storage url {} successfully.".format(name, model_storage_url))
    else:
        if model_storage_url != "":
            click.echo("Push model {} successfully".format(name))
            click.echo("The remote model storage is located at {}".format(model_storage_url))
            click.echo("The local model package is locate at .".format(model_zip))
        else:
            click.echo("Failed to push model {}.".format(name))


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
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    if FedMLModelCards.get_instance().pull_model(name, user, api_key, local_server):
        click.echo("Pull model {} successfully.".format(name))
    else:
        click.echo("Failed to pull model {}.".format(name))


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
def fedml_model_serve(local, name, master_ids, worker_ids, user_id, api_key):
    if local:
        FedMLModelCards.get_instance().local_serve_model(name)
    else:
        if master_ids != "" or worker_ids != "":
            # On-Premise deploy mode
            if master_ids == "" or worker_ids == "":
                click.echo("You must provide both master and worker device id(s).")
                return
            click.echo("Enter the on-premise deployment mode...")
            if user_id == "" and os.environ.get("FEDML_USER_ID", None) is None:
                # Let user enter through command line
                user_id = click.prompt("Please input your user id")
                os.environ["FEDML_USER_ID"] = user_id
            if api_key == "" and os.environ.get("FEDML_API_KEY", None) is None:
                # Let user enter through command line
                api_key = click.prompt("Please input your api key", hide_input=True)
                os.environ["FEDML_API_KEY"] = api_key
            os.environ["FEDML_MODEL_SERVE_MASTER_DEVICE_IDS"] = master_ids
            os.environ["FEDML_MODEL_SERVE_WORKER_DEVICE_IDS"] = worker_ids
            FedMLModelCards.get_instance().serve_model(name)
        else:
            # FedML® Launch deploy mode
            click.echo("Warning: You did not indicate the master device id and worker device id\n\
                       Do you want to use fedml® launch platform to find GPU Resources deploy your model?")
            answer = click.prompt("Please input your answer: (y/n)")
            if answer == "y" or answer == "Y":
                api_key = click.prompt("Please input your api key", hide_input=True)
                # Find the config yaml file in local model cards directory
                yaml_file = FedMLModelCards.get_instance().find_yaml_for_launch(name)
                if yaml_file == "":
                    click.echo("Cannot find the config yaml file for model {}.".format(name))
                    return False
                else:
                    os.chdir(os.path.dirname(yaml_file))    # Set the execution path to the yaml folder
                    from .launch import FedMLLaunchManager
                    version = "dev" #TODO: change to release
                    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
                    if error_code != 0:
                        click.echo("Please check if your API key is valid.")
                        return
                    FedMLLaunchManager.get_instance().set_config_version(version)
                    FedMLLaunchManager.get_instance().api_launch_job(yaml_file, None)
            else:
                click.echo("Please specify both the master device id and worker device ids in the config file.")
                return False