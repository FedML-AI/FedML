#!/usr/bin/env python3
import argparse
import os
import time

import urllib
import urllib.request
import uuid

import yaml
import zipfile


def load_yaml_config(yaml_path):
    """Helper function to load a yaml config file"""
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")


def unzip_file(zip_file, unzip_file_path):
    result = False
    if zipfile.is_zipfile(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall(unzip_file_path)
            result = True

    return result


def retrieve_and_unzip_package(package_name, package_url, saved_package_path):
    package_file_no_extension = str(package_name).split('.')[0]
    local_package_path = "/tmp/fedml_packages"
    try:
        os.mkdir(local_package_path)
    except Exception as e:
        print("unzip package...")
    local_package_path = "/tmp/fedml_packages/" + str(uuid.uuid4())
    try:
        os.mkdir(local_package_path)
    except Exception as e:
        print("unzip the package...")
    local_package_file = local_package_path + "/" + package_name
    urllib.request.urlretrieve(package_url, local_package_file)
    unzip_package_path = local_package_path + "/"
    unzip_file(local_package_file, unzip_package_path)
    unzip_package_path += package_file_no_extension
    copy_cmd = "cp -rf " + unzip_package_path + "/* " + saved_package_path
    print("moving command: " + copy_cmd)
    os.system(copy_cmd)
    return unzip_package_path


def build_fedml_entry_cmd(base_dir):
    cfg_file = os.path.join(base_dir, "conf", "fedml.yaml")
    fedml_config = load_yaml_config(cfg_file)
    source_file = fedml_config["entry_config"]["entry_file"]
    entry_arguments = fedml_config["entry_arguments"]
    entry_cmd = ""
    for entry_key, entry_value in entry_arguments.items():
        entry_cmd = entry_cmd + " --" + entry_key + " " + str(entry_value)
    entry_cmd = "cd " + base_dir + os.path.dirname(source_file) + \
                "; python " + os.path.basename(source_file) + entry_cmd
    print("run cmd " + entry_cmd)
    return entry_cmd


def run_fedml_instance_from_base_package():
    os.system(build_fedml_entry_cmd("/fedml/"))


def run_fedml_instance_from_local_package():
    os.system(build_fedml_entry_cmd("/fedml/fedml-package/"))


if __name__ == "__main__":
    print("Hello, FedML!")

    # parse python script input parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="normal", metavar="N",
                        help="running mode, includes two choices: test and normal.")
    parser.add_argument("--version", type=str, default="latest", metavar="N",
                        help="FedML version (dev / test / latest).")
    parser.add_argument("--package_name", type=str, default="package", metavar="N",
                        help="FedML package name.")
    parser.add_argument("--package_url", type=str, default="https://s3-url", metavar="N",
                        help="FedML package url.")
    args = parser.parse_args()

    if args.mode == "test":
        time.sleep(60)
        exit(0)

    print("Now is downloading the FedML package: " + "name@" + args.package_name + ", url@" + args.package_url)
    retrieve_and_unzip_package(args.package_name, args.package_url, "/fedml/fedml-package/")
    os.system("cp -f /fedml/conf/fedml.yaml /fedml/fedml-package/conf/")

    run_fedml_instance_from_local_package()
