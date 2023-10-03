import os

import click

from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards



def create(name, config_file):
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

    package(name)

def delete(name):
    if FedMLModelCards.get_instance().delete_model(name):
        click.echo("Delete model {} successfully.".format(name))
    else:
        click.echo("Failed to delete model {}.".format(name))


def add_files(name, path):
    if FedMLModelCards.get_instance().add_model_files(name, path):
        click.echo("Add file to model {} successfully.".format(name))
    else:
        click.echo("Failed to add file to model {}.".format(name))


def remove_files(name, file):
    if FedMLModelCards.get_instance().remove_model_files(name, file):
        click.echo("Remove file from model {} successfully.".format(name))
    else:
        click.echo("Failed to remove file from model {}.".format(name))


def list_models(name):
    models = FedMLModelCards.get_instance().list_models(name)
    if name == "*":
        if len(models) <= 0:
            click.echo("Model list is empty.")
        else:
            click.echo("-------------------------")
            click.echo("Model Name")
            click.echo("-------------------------")
            for model_item in models:
                click.echo(model_item)
            click.echo("-------------------------")
    else:
        if len(models) <= 0:
            click.echo("Cannot locate model {}.".format(name))

def list_remote(name, user, api_key):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    model_query_result = FedMLModelCards.get_instance().list_models(name, user, api_key)
    if model_query_result is None or model_query_result.model_list is None or len(model_query_result.model_list) <= 0:
        click.echo("Model list is empty.")
    else:
        click.echo("Found {} models:".format(len(model_query_result.model_list)))
        index = 1
        for model_item in model_query_result.model_list:
            model_item.show("{}. ".format(index))
            index += 1
        click.echo("List model {} successfully.".format(name))


def package(name):
    model_zip = FedMLModelCards.get_instance().build_model(name)
    if model_zip != "":
        click.echo("Build model package {} successfully".format(name))
        click.echo("The local package is located at {}.".format(model_zip))
    else:
        click.echo("Failed to build model {}.".format(name))


def push(name, model_storage_url, model_net_url, user, api_key):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    model_is_from_open = True if model_storage_url is not None and model_storage_url != "" else False
    model_storage_url, model_zip = FedMLModelCards.get_instance().push_model(name, user, api_key,
                                                                             model_storage_url=model_storage_url,
                                                                             model_net_url=model_net_url)
    if model_is_from_open:
        click.echo("Push model {} with model storage url {} successfully.".format(name, model_storage_url))
    else:
        if model_storage_url != "":
            click.echo("Push model {} successfully".format(name))
            click.echo("The remote model storage is located at {}".format(model_storage_url))
            click.echo("The local model package is locate at .".format(model_zip))
        else:
            click.echo("Failed to push model {}.".format(name))


def pull(name, user, api_key):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    if FedMLModelCards.get_instance().pull_model(name, user, api_key):
        click.echo("Pull model {} successfully.".format(name))
    else:
        click.echo("Failed to pull model {}.".format(name))


def deploy(local, name, master_ids, worker_ids, user_id, api_key, config_file):
    if local:
        FedMLModelCards.get_instance().local_serve_model(name, config_file)
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
            FedMLModelCards.get_instance().serve_model(name, config_file)
        else:
            # FedML® Launch deploy mode
            click.echo("Warning: You did not indicate the master device id and worker device id\n\
                       Do you want to use fedml® launch platform to find GPU Resources deploy your model?")
            answer = click.prompt("Please input your answer: (y/n)")
            if answer == "y" or answer == "Y":
                from .launch import FedMLLaunchManager
                api_key = FedMLLaunchManager.get_api_key()
                # Find the config yaml file in local model cards directory ~/fedml-model-client/fedml/models
                yaml_file = FedMLModelCards.get_instance().prepare_yaml_for_launch(name)
                if yaml_file == "":
                    click.echo("Cannot find the config yaml file for model {}.".format(name))
                    return False
                else:
                    os.chdir(os.path.dirname(yaml_file))    # Set the execution path to the yaml folder
                    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key)
                    if error_code != 0:
                        click.echo("Please check if your API key is valid.")
                        return
                    FedMLLaunchManager.get_instance().api_launch_job(yaml_file, None)
            else:
                click.echo("Please specify both the master device id and worker device ids in the config file.")
                return False


def info(name):
    inference_output_url, model_version, model_metadata, model_config = FedMLModelCards.get_instance().query_model(name)
    if inference_output_url != "":
        click.echo("Query model {} successfully.".format(name))
        click.echo("infer url: {}.".format(inference_output_url))
        click.echo("model version: {}.".format(model_version))
        click.echo("model metadata: {}.".format(model_metadata))
        click.echo("model config: {}.".format(model_config))
    else:
        if name is None:
            print("please specifiy the model name")
        else:
            click.echo("Failed to query model {}.".format(name))


def run(name, data):
    infer_out_json = FedMLModelCards.get_instance().inference_model(name, data)
    if infer_out_json != "":
        click.echo("Inference model {} successfully.".format(name))
        click.echo("Result: {}.".format(infer_out_json))
    else:
        click.echo("Failed to inference model {}.".format(name))

