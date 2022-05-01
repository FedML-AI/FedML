# Copyright 2022, FedML.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Arguments."""

import argparse
from os import path

import yaml


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )

    # default arguments
    parser.add_argument("--run_id", type=str, default="0")

    # default arguments
    parser.add_argument("--rank", type=int, default=0)

    args = parser.parse_args()
    return args


class Arguments:
    """Argument class which contains all arguments from yaml config and constructs additional arguments"""

    def __init__(self, cmd_args, training_type=None, comm_backend=None):
        # set the command line arguments
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)

        self.get_default_yaml_config(
            cmd_args, training_type, comm_backend
        )

    def load_yaml_config(self, yaml_path):
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    def get_default_yaml_config(self, cmd_args, training_type=None, comm_backend=None):
        path_current_file = path.abspath(path.dirname(__file__))
        if training_type == "simulation" and comm_backend == "single_process":
            config_file = path.join(path_current_file, "config/simulation_sp/fedml_config.yaml")
            cmd_args.yaml_config_file = config_file
        elif training_type == "simulation" and comm_backend == "MPI":
            config_file = path.join(
                path_current_file, "config/simulaton_mpi/fedml_config.yaml"
            )
            cmd_args.yaml_config_file = config_file
        elif training_type == "cross_silo":
            pass
        elif training_type == "cross_device":
            pass
        else:
            pass

        self.yaml_paths = [cmd_args.yaml_config_file]
        # Load all arguments from yaml config
        # https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started
        configuration = self.load_yaml_config(cmd_args.yaml_config_file)

        # Override class attributes from current yaml config
        for _, param_family in configuration.items():
            for key, val in param_family.items():
                setattr(self, key, val)

        path_current_file = path.abspath(path.dirname(__file__))
        if training_type == "simulation" and comm_backend == "single_process":
            pass
        elif training_type == "simulation" and comm_backend == "MPI":
            self.gpu_mapping_file = path.join(
                path_current_file, "config/simulaton_mpi/gpu_mapping.yaml"
            )
        elif training_type == "cross_silo":
            pass
        elif training_type == "cross_device":
            pass
        else:
            pass
        return configuration


def load_arguments(training_type=None, comm_backend=None):
    cmd_args = add_args()
    # Load all arguments from YAML config file
    args = Arguments(cmd_args, training_type, comm_backend)
    return args
