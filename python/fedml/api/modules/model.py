import os

import click

import fedml.api
import shutil
import yaml

import urllib.request
import zipfile

from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.api.modules.utils import fedml_login
from fedml.computing.scheduler.comm_utils.security_utils import get_api_key, save_api_key


def create(name: str, model: str = None, model_config: str = None) -> bool:
    if model is not None:
        if model.startswith("hf:"):
            if create_from_hf(name, model[3:]):
                return True
            else:
                return False
        else:
            # TODO: Support arbitrary model creation from GitHub / Nexus AI Job Store
            click.echo("Model {} is not supported yet.".format(model))
            return False

    if model_config is None:
        # Just create a model folder
        if FedMLModelCards.get_instance().create_model(name):
            click.echo("Create model {} successfully.".format(name))
            package(name)
            return True
        else:
            click.echo("Failed to create model {}.".format(name))
            return False
    else:
        # Adding related workspace codes to the model folder
        if FedMLModelCards.get_instance().create_model_use_config(name, model_config):
            click.echo("Create model {} using config successfully.".format(name))
            package(name)
            return True
        else:
            click.echo("Failed to create model {} using config file {}.".format(name, model_config))
            return False


def create_from_hf(name: str, model: str = None) -> bool:
    """
    Pull template from s3
    [Deprecated Solution] Packaging the template to whl file, i.e.
    # setup.py
    package_data={
        "fedml.serving": ["**/*.sh", "**/*.yaml", "**/*.json", "**/*.txt", "**/*.md", "**/*.jinja"],
    },
    # fedml/api/modules/model.py
    hf_templ_fd_src = os.path.join(os.path.dirname(__file__), "..", "..", "serving", "templates", "hf_template")
    """
    templ_zip_url = "https://fedml-deploy-template.s3.us-west-2.amazonaws.com/hf_template.zip"
    hf_templ_parent_fd = os.path.join(os.path.expanduser("~"), ".fedml", "fedml-model-client", "fedml")
    hf_templ_fd = os.path.join(hf_templ_parent_fd, "hf_template")

    if os.path.exists(hf_templ_fd):
        shutil.rmtree(hf_templ_fd)
    os.makedirs(hf_templ_fd)

    print(f"Downloading the template from {templ_zip_url} ... to {hf_templ_parent_fd}")
    urllib.request.urlretrieve(templ_zip_url, os.path.join(hf_templ_parent_fd, "hf_template.zip"))

    with zipfile.ZipFile(os.path.join(hf_templ_parent_fd, "hf_template.zip"), 'r') as zip_ref:
        zip_ref.extractall(hf_templ_parent_fd)

    # Fulfill the template to a new model card that is deployable
    dst_parent_fd = os.path.join(os.path.expanduser("~"), ".fedml",
                                 "fedml-model-client", "fedml", "hf_model_from_template")
    if not os.path.exists(dst_parent_fd):
        os.makedirs(dst_parent_fd)

    dst_fd = os.path.join(dst_parent_fd, name)
    if os.path.exists(dst_fd):
        shutil.rmtree(dst_fd)

    print(f"Copying the template from {hf_templ_fd} ... to {dst_fd}")
    shutil.copytree(hf_templ_fd, dst_fd)
    
    config_file = os.path.join(dst_fd, "config.yaml")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        config["environment_variables"]["MODEL_NAME_OR_PATH"] = model   # Change the hf_model_name in the config.yaml
    with open(config_file, 'w') as file:
        yaml.dump(config, file)
    
    res = create(name, model_config=config_file)
    if res:
        print(f"Model source code (generated from template {hf_templ_fd}) is located at {dst_fd}")
    return res


def delete(name: str, local: bool = True) -> bool:
    if local:
        if FedMLModelCards.get_instance().delete_model(name):
            click.echo("Delete model {} successfully.".format(name))
            return True
        else:
            click.echo("Failed to delete model {}.".format(name))
            return False
    else:
        # TODO: Delete model from server
        pass


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


def list_models(name: str = "*", local: bool = True) -> any:
    if local:
        models = FedMLModelCards.get_instance().list_models(name)
        if name == "*":
            if len(models) <= 0:
                click.echo("Model list is empty.")
                return []
            else:
                click.echo("-------------------------")
                click.echo("Model Name")
                click.echo("-------------------------")
                for model_item in models:
                    click.echo(model_item)
                click.echo("-------------------------")
                return models
        else:
            if len(models) <= 0:
                click.echo("Cannot locate model {}.".format(name))
                return []
            else:
                click.echo("Found model {}.".format(name))
                return models
    else:
        api_key = get_api_key()
        if api_key == "":
            click.echo('''
            Please use one of the ways below to login first:
            (1) CLI: `fedml login $api_key`
            (2) API: fedml.api.fedml_login(api_key=$api_key)
            ''')
            return None

        model_query_result = FedMLModelCards.get_instance().list_models(name, api_key)
        if model_query_result is None or model_query_result.model_list is None or len(
                model_query_result.model_list) <= 0:
            click.echo("Model list is empty.")
            return []
        else:
            click.echo("Found {} models:".format(len(model_query_result.model_list)))
            index = 1
            for model_item in model_query_result.model_list:
                model_item.show("{}. ".format(index))
                index += 1
            click.echo("List model {} successfully.".format(name))
            return model_query_result.model_list


def package(name: str) -> str:
    model_zip = FedMLModelCards.get_instance().build_model(name)
    if model_zip != "":
        click.echo("Build model package {} successfully".format(name))
        click.echo("The local package is located at {}.".format(model_zip))
        return model_zip
    else:
        click.echo("Failed to build model {}.".format(name))
        return ""


def push(name: str, model_storage_url: str = None, api_key: str = None, tag_names: list = None,
          model_id: int = None, model_version: str = None) -> bool:
    if api_key is None or str(api_key).strip() == "":
        api_key = get_api_key()
    if api_key == "":
        click.echo('''
        Please use one of the ways below to login first:
        (1) CLI: `fedml login $api_key`
        (2) API: fedml.api.fedml_login(api_key=$api_key)
        ''')
        return False

    model_is_from_open = True if model_storage_url is not None and model_storage_url != "" else False

    model_storage_url, model_zip = FedMLModelCards.get_instance().push_model(name, "", api_key,
                                                                             model_storage_url=model_storage_url,
                                                                             model_net_url="",
                                                                             tag_names=tag_names,
                                                                             model_id=model_id,
                                                                             model_version=model_version)
    if model_is_from_open:
        click.echo("Push model {} with model storage url {} successfully.".format(name, model_storage_url))
        return True
    else:
        if model_storage_url != "":
            click.echo("Push model {} successfully".format(name))
            click.echo("The remote model storage is located at {}".format(model_storage_url))
            click.echo("The local model package is locate at .".format(model_zip))
            return True
        else:
            click.echo("Failed to push model {}.".format(name))
            return False


def pull(name: str) -> any:
    api_key = get_api_key()
    if api_key == "":
        click.echo('''
        Please use one of the ways below to login first:
        (1) CLI: `fedml login $api_key`
        (2) API: fedml.api.fedml_login(api_key=$api_key)
        ''')
        return None

    res = FedMLModelCards.get_instance().pull_model(name, "", api_key)
    if res != "":
        click.echo("Pull model {} successfully.".format(name))
    else:
        click.echo("Failed to pull model {}.".format(name))
    return res


def deploy(name: str, endpoint_name: str = "", endpoint_id: str = None, local: bool = False, master_ids: str = None,
           worker_ids: str = None, use_remote: bool = False) -> bool:
    if local:
        return FedMLModelCards.get_instance().local_serve_model(name)
    else:
        if master_ids is not None or worker_ids is not None:
            # On-Premise deploy mode
            if master_ids is None or worker_ids is None:
                click.echo("You must provide both master and worker device id(s).")
                return False
            click.echo("Enter the on-premise deployment mode...")

            return FedMLModelCards.get_instance().serve_model_on_premise(
                name, endpoint_name, master_ids, worker_ids, use_remote, endpoint_id)
        else:
            # FedML® Launch deploy mode
            click.echo("Warning: You did not indicate the master device id and worker device id\n\
                       Do you want to use FedML® Nexus AI Platform to find GPU Resources deploy your model?")
            answer = click.prompt("Please input your answer: (y/n)")
            if answer == "y" or answer == "Y":
                api_key = get_api_key()
                if api_key == "":
                    click.echo('''
                            Please use one of the ways below to login first:
                            (1) CLI: `fedml login $api_key`
                            (2) API: fedml.api.fedml_login(api_key=$api_key)
                            ''')
                    return False

                yaml_file = FedMLModelCards.get_instance().prepare_yaml_for_launch(name)
                if yaml_file == "":
                    click.echo("Cannot find the config yaml file for model {}.".format(name))
                    return False
                else:
                    saved_original_path = os.getcwd()
                    os.chdir(os.path.dirname(yaml_file))  # Set the execution path to the yaml folder
                    error_code, _ = fedml_login(api_key=api_key)
                    if error_code != 0:
                        click.echo("Please check if your API key is valid.")
                        return False

                    fedml.api.launch_job(yaml_file)
                    os.remove(yaml_file)
                    os.chdir(saved_original_path)
                    return True
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


def run(endpoint_id: str, json_string: str) -> bool:
    api_key = get_api_key()
    if api_key == "":
        click.echo('''
                Please use one of the ways below to login first:
                (1) CLI: `fedml login $api_key`
                (2) API: fedml.api.fedml_login(api_key=$api_key)
                ''')
        return False
    infer_out_json = FedMLModelCards.get_instance().endpoint_inference_api(api_key, endpoint_id, json_string)
    if infer_out_json != "":
        click.echo("Model run successfully.")
        click.echo("Result: {}.".format(infer_out_json))
        return True
    else:
        click.echo("Failed to run model.")
        return False


def delete_endpoint(endpoint_id: str) -> bool:
    api_key = get_api_key()
    if api_key == "":
        click.echo('''
                Please use one of the ways below to login first:
                (1) CLI: `fedml login $api_key`
                (2) API: fedml.api.fedml_login(api_key=$api_key)
                ''')
        return False
    if FedMLModelCards.get_instance().delete_endpoint(api_key, endpoint_id):
        click.echo("Model endpoint {} deleted successfully.".format(endpoint_id))
        return True
    else:
        click.echo("Failed to delete model endpoint {}.".format(endpoint_id))
        return False
