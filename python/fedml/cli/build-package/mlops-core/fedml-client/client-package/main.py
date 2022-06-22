#!/usr/bin/env python3
import argparse
import os
import shutil
import stat
import time
import traceback

import urllib
import urllib.request
import uuid

import yaml
import zipfile

is_local_test = False
is_mac = False
if is_mac:
    fedml_test_dir = os.path.join("Users", "alexliang", "fedml-test")
else:
    fedml_test_dir = os.path.join("/", "tmp")


def load_yaml_config(yaml_path):
    """Helper function to load a yaml config file"""
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")


def generate_yaml_doc(run_config_object, yaml_file):
    try:
        file = open(yaml_file, "w", encoding="utf-8")
        yaml.dump(run_config_object, file)
        file.close()
    except Exception as e:
        print("Generate yaml file.")


def unzip_file(zip_file, unzip_file_path):
    result = False
    if zipfile.is_zipfile(zip_file):
        with zipfile.ZipFile(zip_file, "r") as zipf:
            zipf.extractall(unzip_file_path)
            result = True

    return result


def retrieve_and_unzip_package(package_name, package_url, saved_package_path):
    package_file_no_extension = str(package_name).split(".")[0]
    local_package_path = os.path.join(fedml_test_dir, "fedml_packages")
    try:
        os.makedirs(local_package_path)
    except Exception as e:
        print("unzip package...")
    local_package_path = os.path.join(fedml_test_dir, "fedml_packages", str(uuid.uuid4()))
    try:
        os.makedirs(local_package_path)
    except Exception as e:
        print("unzip the package...")
    local_package_file = os.path.join(local_package_path, package_name)
    urllib.request.urlretrieve(package_url, local_package_file)
    unzip_package_path = local_package_path
    unzip_file(local_package_file, unzip_package_path)
    unzip_package_path = os.path.join(unzip_package_path, package_file_no_extension)
    try:
        shutil.copytree(unzip_package_path, saved_package_path, ignore=True)
    except Exception as e:
        pass
    return unzip_package_path


def build_dynamic_args(run_config, base_dir):
    package_cfg_file = os.path.join(base_dir, "conf", "fedml.yaml")
    print("package_conf_file " + package_cfg_file)
    package_config = load_yaml_config(package_cfg_file)
    fedml_conf_file = package_config["entry_config"]["conf_file"]
    print("fedml_conf_file:" + fedml_conf_file)
    fedml_conf_path = os.path.join(base_dir, "fedml", fedml_conf_file)
    fedml_conf_object = load_yaml_config(fedml_conf_path)
    package_dynamic_args = package_config["dynamic_args"]
    fedml_conf_object["comm_args"]["mqtt_config_path"] = package_dynamic_args[
        "mqtt_config_path"
    ]
    fedml_conf_object["comm_args"]["s3_config_path"] = package_dynamic_args[
        "s3_config_path"
    ]
    fedml_conf_object["common_args"]["using_mlops"] = True
    fedml_conf_object["train_args"]["run_id"] = package_dynamic_args["run_id"]
    fedml_conf_object["train_args"]["client_id_list"] = package_dynamic_args[
        "client_id_list"
    ]
    fedml_conf_object["train_args"]["client_num_in_total"] = int(
        package_dynamic_args["client_num_in_total"]
    )
    fedml_conf_object["train_args"]["client_num_per_round"] = int(
        package_dynamic_args["client_num_in_total"]
    )
    fedml_conf_object["device_args"]["worker_num"] = int(
        package_dynamic_args["client_num_in_total"]
    )
    fedml_conf_object["data_args"]["data_cache_dir"] = package_dynamic_args[
        "data_cache_dir"
    ]
    fedml_conf_object["tracking_args"]["log_file_dir"] = package_dynamic_args[
        "log_file_dir"
    ]
    fedml_conf_object["tracking_args"]["log_server_url"] = package_dynamic_args[
        "log_server_url"
    ]
    bootstrap_script_file = fedml_conf_object["environment_args"]["bootstrap"]
    bootstrap_script_path = os.path.join(
        base_dir, "fedml", "config", os.path.basename(bootstrap_script_file)
    )
    try:
        os.makedirs(package_dynamic_args["data_cache_dir"])
    except Exception as e:
        pass
    fedml_dynamic_args = fedml_conf_object.get("dynamic_args", None)
    if fedml_dynamic_args is not None:
        for entry_key, entry_value in package_dynamic_args.items():
            fedml_dynamic_args[entry_key] = entry_value

    generate_yaml_doc(fedml_conf_object, fedml_conf_path)
    try:
        bootstrap_stat = os.stat(bootstrap_script_path)
        os.chmod(bootstrap_script_path, bootstrap_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.system(bootstrap_script_path)
    except Exception as e:
        print("Exception at {}".format(traceback.format_exc()))


def build_fedml_entry_cmd(base_dir):
    package_cfg_file = os.path.join(base_dir, "conf", "fedml.yaml")
    package_config = load_yaml_config(package_cfg_file)
    source_file = package_config["entry_config"]["entry_file"]
    fedml_conf_file = package_config["entry_config"]["conf_file"]
    package_dynamic_args = package_config["dynamic_args"]
    entry_cmd = (
        " --cf " + fedml_conf_file + " --rank " + str(package_dynamic_args["rank"])
    )
    if is_local_test:
        entry_cmd = (
            "cd "
            + base_dir
            + os.path.dirname(source_file)
            + "; python3 "
            + os.path.basename(source_file)
            + entry_cmd
        )
    else:
        entry_cmd = (
            "cd "
            + base_dir
            + os.path.dirname(source_file)
            + "; python "
            + os.path.basename(source_file)
            + entry_cmd
        )
    print("source_file " + source_file)
    build_dynamic_args(base_dir)
    print("run cmd " + entry_cmd)
    return entry_cmd


def run_fedml_instance_from_base_package():
    if is_local_test:
        fedml_dir = os.path.join(fedml_test_dir, "fedml")
    else:
        fedml_dir = os.path.join("/", "fedml")
    os.system(build_fedml_entry_cmd(fedml_dir))


def run_fedml_instance_from_local_package():
    if is_local_test:
        fedml_package_local_dir = os.path.join(fedml_test_dir, "fedml", "fedml-package")
    else:
        fedml_package_local_dir = os.path.join("/", "fedml", "fedml-package")
    os.system(build_fedml_entry_cmd(fedml_package_local_dir))


if __name__ == "__main__":
    print("Hello, FedML!")

    # parse python script input parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        metavar="N",
        help="running mode, includes two choices: test and normal.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="latest",
        metavar="N",
        help="FedML version (dev / test / latest).",
    )
    parser.add_argument(
        "--package_name",
        type=str,
        default="package",
        metavar="N",
        help="FedML package name.",
    )
    parser.add_argument(
        "--package_url",
        type=str,
        default="https://s3-url",
        metavar="N",
        help="FedML package url.",
    )
    args = parser.parse_args()

    if args.mode == "test":
        time.sleep(60)
        exit(0)

    print(
        "Now is downloading the FedML package: "
        + "name@"
        + args.package_name
        + ", url@"
        + args.package_url
    )
    if is_local_test:
        fedml_package_local_dir = os.path.join(fedml_test_dir, "fedml", "fedml-package")
        fedml_conf_dir = os.path.join(fedml_test_dir, "fedml", "conf")
        fedml_conf_file = os.path.join(fedml_conf_dir, "fedml.yaml")
        fedml_package_conf_dir = os.path.join(fedml_test_dir, "fedml", "fedml-package", "conf")
        try:
            os.makedirs(fedml_package_local_dir)
        except Exception as e:
            pass
        try:
            os.makedirs(fedml_conf_dir)
        except Exception as e:
            pass
        try:
            os.makedirs(fedml_package_conf_dir)
        except Exception as e:
            pass
        try:
            shutil.copyfile(os.path.join(os.getcwd(), "conf", "fedml.yaml"),
                            os.path.join(fedml_conf_dir, "fedml.yaml"))
        except Exception as e:
            pass
    else:
        fedml_package_local_dir = os.path.join("/", "fedml", "fedml-package")
        fedml_conf_dir = os.path.join("/", "fedml", "conf")
        fedml_conf_file = os.path.join(fedml_conf_dir, "fedml.yaml")
        fedml_package_conf_dir = os.path.join("/", "fedml", "fedml-package", "conf")
        try:
            os.makedirs(fedml_package_local_dir)
        except Exception as e:
            pass
        try:
            os.makedirs(fedml_conf_dir)
        except Exception as e:
            pass
        try:
            os.makedirs(fedml_package_conf_dir)
        except Exception as e:
            pass
    retrieve_and_unzip_package(
        args.package_name, args.package_url, fedml_package_local_dir
    )

    try:
        shutil.copyfile(fedml_conf_file, os.path.join(fedml_package_conf_dir, "fedml.yaml"))
    except Exception as e:
        pass

    run_fedml_instance_from_local_package()
