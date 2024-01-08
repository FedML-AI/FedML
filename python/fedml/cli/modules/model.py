import click

import fedml.api
from fedml.cli.modules.utils import OrderedGroup


@click.group("model", cls=OrderedGroup)
@click.help_option("--help", "-h")
def fedml_model():
    """
     FedML Model CLI will help you manage the model cards, whether it is in local environment or at FedML
     Nexus AI platform. It also helps you to deploy the model cards to different devices, and manage the endpoints
     that created from model cards.
    """
    pass


@fedml_model.command("create", help="Create a model card in local environment.")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release"
)
@click.option(
    "--name", "-n", type=str, help="Model Card name.", required=True
)
@click.option(
    "--model", "-m", type=str, default=None, help="Indicate a pre-built model from Hugging Face or GitHub."
                                                  " e.g. hf:EleutherAI/pythia-70m."
)
@click.option(
    "--model_config", "-cf", default=None, type=str, help="Yaml file path that will be used to create a"
                                                          " new model card.",
)
def fedml_model_create(version, name, model, model_config):
    fedml.set_env_version(version)
    if name is None:
        click.echo("You must provide a model name (use -n option).")
        return
    fedml.api.model_create(name, model, model_config)


@fedml_model.command("push", help="Push a model card (local or S3) to remote.")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release"
)
@click.option(
    "--name", "-n", type=str, help="Model card name.", required=True
)
@click.option(
    "--model_storage_url", "-s", type=str, help="A S3 address to the model card zip file.",
)
@click.option(
    "--api_key", "-k", type=str, help="API key for the Nexus AI Platform.",
)
@click.option(
    "--tag_names", "-t", type=str, default=None, help="Tag for the model card.",
)
@click.option(
    "--model_id", "-i", type=int, default=None, help="Model card version."
)
@click.option(
    "--model_version", "-m", type=str, default=None, help="Model card version."
)
def fedml_model_push(name, model_storage_url, version, api_key, tag_names, model_id, model_version):
    fedml.set_env_version(version)
    if name is None:
        click.echo("You must provide a model name (use -n option).")
        return
    fedml.api.model_push(name, model_storage_url, api_key, tag_names, model_id, model_version)


@fedml_model.command("deploy", help="Deploy model to the local | on-premise | GPU Cloud.")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release"
)
@click.option(
    "--name", "-n", type=str, help="Model Card Name.",
)
@click.option(
    "--endpoint_name", "-e", type=str, default="", help="Endpoint name."
)
@click.option(
    "--endpoint_id", "-i", type=str, default="", help="Endpoint id."
)
@click.option(
    "--local", "-l", default=False, is_flag=True, help="Deploy model locally.",
)
@click.option(
    "--master_ids", "-m", type=str, default=None, help=" Device Id(s) for on-premise master node(s)."
                                                     " Please indicate master device id(s), seperated with ','"
)
@click.option(
    "--worker_ids", "-w", type=str, default=None, help=" Device Id(s) for on-premise worker node(s)."
                                                     " Please indicate worker device id(s), seperated with ','"
)
@click.option(
    "--use_remote", "-r", default=False, is_flag=True, help="Use the model card on the Nexus AI Platform. Default is"
                                                            " False, which means use the model card in local."
)
@click.option(
    "--delete", "-d", type=str, default="", help="Delete a model endpoint using endpoint id."
)
def fedml_model_deploy(version, local, name, endpoint_name, endpoint_id, master_ids, worker_ids, use_remote, delete):
    fedml.set_env_version(version)
    if delete != "":
        click.confirm(
            "Are you sure to delete the model endpoint: {}".format(delete),
            abort=True,
        )
        fedml.api.endpoint_delete(delete)
        return
    if name is None:
        click.echo("You must provide a model name (use -n option).")
        return
    fedml.api.model_deploy(name, endpoint_name, endpoint_id, local, master_ids, worker_ids, use_remote)


@fedml_model.command("run", help="Request a model inference endpoint.")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release"
)
@click.option(
    "--endpoint", "-e", type=str, help="Model endpoint id.",
)
@click.argument("JSON_STRING", type=str)
def fedml_model_run(endpoint, version, json_string):
    fedml.set_env_version(version)
    if endpoint is None:
        click.echo("You must provide a model endpoint id (use -e option).")
        return
    fedml.api.model_run(endpoint, json_string)


@fedml_model.command("pull", help="Pull a model card from Nexus AI Platform to local.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="Model card name.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
def fedml_model_pull(name, version):
    fedml.set_env_version(version)
    if name is None:
        click.echo("You must provide a model name (use -n option).")
        return
    fedml.api.model_pull(name)


@fedml_model.command("list", help="List model card(s) at local environment or Nexus AI Platform.")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, default="*", help=
    '''
    Model card(s) name. "*" means all model cards. To select multiple model cards, use "," to separate them.
    e.g. "model1,model2".
    '''
)
@click.option(
    "--local", "-l", default=False, is_flag=True, help="List model locally.",
)
def fedml_model_list(version, name, local):
    fedml.set_env_version(version)
    if name is None:
        click.echo("You must provide a model name (use -n option).")
        return

    fedml.api.model_list(name, local)


@fedml_model.command("delete", help="Delete a local or remote model card.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="Model card name",
)
@click.option(
    "--local", "-l", default=False, is_flag=True, help="Delete the model card in local environment.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
def fedml_model_delete(name, local, version):
    fedml.set_env_version(version)
    if name is None:
        click.echo("You must provide a model name (use -n option).")
        return
    fedml.api.model_delete(name, local)


@fedml_model.command("package", help="Pakcage a local or remote model card. So that can be uploaded through UI\
                     to Nexus AI Platform.")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, default=None, help=
    '''
    Model card(s) name. "*" means all model cards. To select multiple model cards, use "," to separate them.
    e.g. "model1,model2".
    '''
)
def fedml_model_pacakge(name, version):
    fedml.api.model_package(name)