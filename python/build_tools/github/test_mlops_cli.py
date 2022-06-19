import argparse
import json
import os.path
import time
import traceback
from os.path import expanduser

import psutil
import requests
import yaml
from fedml.cli.comm_utils.mqtt_manager import MqttManager
from fedml.core.mlops.mlops_configs import MLOpsConfigs


def load_yaml_config(yaml_path):
    """Helper function to load a yaml config file"""
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")


def test_is_ok(test_run_id, test_edge_id, test_function, ok_message):
    test_ok = False
    wait_count = 0
    log_file_path = os.path.join(args.log_dir,
                                 "fedml-run-{}-edge-{}.log".format(str(test_run_id), str(test_edge_id)))
    log_file = open(log_file_path, "r")
    while True:
        if wait_count >= 30:
            break
        log_line = log_file.readline()
        if not log_line:
            wait_count += 1
            time.sleep(3)
            continue
        if log_line.find(ok_message) != -1:
            test_ok = True
            break

    log_file.close()
    if test_ok:
        print("{} with successful status.".format(test_function.__name__))
    else:
        print("{} with failure status.".format(test_function.__name__))


def load_edge_infos():
    home_dir = expanduser("~")
    local_pkg_data_dir = os.path.join(home_dir, "fedml-client", "fedml", "data")
    edge_info_file = os.path.join(local_pkg_data_dir, "edge_infos.yaml")
    edge_info_file_handle = load_yaml_config(edge_info_file)
    return edge_info_file_handle["unique_device_id"], edge_info_file_handle["edge_id"]


def test_login_with_start_run_by_sending_client_agent_msg(args):
    test_run_id = 873
    _, test_edge_id = load_edge_infos()
    start_train_topic = "flserver_agent/{}/start_train".format(str(test_edge_id))
    start_train_msg = {
        "edges": [
            {
                "device_id": "@0x9801a7a55e85.MacOS",
                "os_type": "MacOS",
                "id": test_edge_id
            }
        ],
        "starttime": 1651664950745,
        "url": "http://fedml-server-agent-svc.fedml-aggregator-dev.svc.cluster.local:5001/api/start_run",
        "edgeids": [
            test_edge_id
        ],
        "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA1LCJhY2NvdW50IjoiYWxleC5saWFuZzIiLCJsb2dpblRpbWUiOiIxNjUxNjY0NzM1NDA1IiwiZXhwIjowfQ.vtgrNBhcCVy_d2oI9NeSFcwPxHHSWvtoZS_k-_SLAyU",
        "urls": [],
        "userids": [
            "105"
        ],
        "name": "eye_rain",
        "runId": test_run_id,
        "id": test_run_id,
        "projectid": "121",
        "run_config": {
            "configName": "test-new-open",
            "userId": 105,
            "model_config": {},
            "packages_config": {
                "server": "server-package.zip",
                "linuxClient": "client-package.zip",
                "serverUrl": "https://fedml.s3.us-west-1.amazonaws.com/1651664769220server-package.zip",
                "linuxClientUrl": "https://fedml.s3.us-west-1.amazonaws.com/1651664778846client-package.zip",
                "androidClient": "",
                "androidClientUrl": "",
                "androidClientVersion": "0"
            },
            "data_config": {
                "privateLocalData": "",
                "syntheticData": "",
                "syntheticDataUrl": ""
            },
            "parameters": {
                "model_args": {
                    "model_file_cache_folder": "./model_file_cache",
                    "model": "lr",
                    "global_model_file_path": "./model_file_cache/global_model.pt"
                },
                "device_args": {
                    "worker_num": 2,
                    "using_gpu": False,
                    "gpu_mapping_key": "mapping_default",
                    "gpu_mapping_file": "config/gpu_mapping.yaml"
                },
                "comm_args": {
                    "s3_config_path": "config/s3_config.yaml",
                    "backend": "MQTT_S3",
                    "mqtt_config_path": "config/mqtt_config.yaml"
                },
                "train_args": {
                    "batch_size": 10,
                    "weight_decay": 0.001,
                    "client_num_per_round": 2,
                    "client_num_in_total": 2,
                    "comm_round": 50,
                    "client_optimizer": "sgd",
                    "client_id_list": "[1, 2]",
                    "epochs": 1,
                    "learning_rate": 0.03,
                    "federated_optimizer": "FedAvg"
                },
                "environment_args": {
                    "bootstrap": "config/bootstrap.sh"
                },
                "validation_args": {
                    "frequency_of_the_test": 1
                },
                "common_args": {
                    "random_seed": 0,
                    "training_type": "cross_silo",
                    "using_mlops": False
                },
                "data_args": {
                    "partition_method": "hetero",
                    "partition_alpha": 0.5,
                    "dataset": "mnist",
                    "data_cache_dir": "../../../../data/mnist"
                },
                "tracking_args": {
                    "wandb_project": "fedml",
                    "wandb_name": "fedml_torch_fedavg_mnist_lr",
                    "wandb_key": "ee0b5f53d949c84cee7decbe7a629e63fb2f8408",
                    "enable_wandb": False,
                    "log_file_dir": "./log"
                }
            }
        },
        "timestamp": "1651664950759"
    }

    mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
    mqtt_mgr = MqttManager(
        mqtt_config["BROKER_HOST"],
        mqtt_config["BROKER_PORT"],
        mqtt_config["MQTT_USER"],
        mqtt_config["MQTT_PWD"],
        mqtt_config["MQTT_KEEPALIVE"],
        "login-test",
    )

    mqtt_mgr.send_message_json(start_train_topic, json.dumps(start_train_msg))

    time.sleep(3)

    test_is_ok(test_run_id, test_edge_id,
               test_login_with_start_run_by_sending_client_agent_msg, "Connection is ready!")


def test_login_with_stop_run_by_sending_client_agent_msg(args):
    test_run_id = 873
    _, test_edge_id = load_edge_infos()
    stop_train_topic = "flserver_agent/{}/stop_train".format(str(test_edge_id))
    stop_train_msg = {
        "edgeids": [
            test_edge_id
        ],
        "runId": test_run_id,
    }

    mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
    mqtt_mgr = MqttManager(
        mqtt_config["BROKER_HOST"],
        mqtt_config["BROKER_PORT"],
        mqtt_config["MQTT_USER"],
        mqtt_config["MQTT_PWD"],
        mqtt_config["MQTT_KEEPALIVE"],
        "login-test",
    )

    mqtt_mgr.send_message_json(stop_train_topic, json.dumps(stop_train_msg))

    time.sleep(5)

    all_is_killed = True
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=['pid', 'name', "cmdline"])
            for cmd in pinfo["cmdline"]:
                if str(cmd).find("fedml_config.yaml") != -1:
                    all_is_killed = False
        except Exception as e:
            pass

    if all_is_killed:
        print("{} with successful status.".format(test_login_with_stop_run_by_sending_client_agent_msg.__name__))
    else:
        print("{} with failure status.".format(test_login_with_stop_run_by_sending_client_agent_msg.__name__))


def send_request_to_server_agent(args, request_json):
    url = "https://open.fedml.ai:5001/api/start_run"
    if hasattr(args, "config_version") and args.config_version is not None:
        # Setup config url based on selected version.
        if args.config_version == "release":
            url = "https://open.fedml.ai:5001/api/start_run"
        elif args.config_version == "test":
            url = "http://open-test.fedml.ai:5001/api/start_run"
        elif args.config_version == "dev":
            url = "http://open-dev.fedml.ai:5001/api/start_run"
        elif args.config_version == "local":
            url = "http://localhost:5001/api/start_run"

    json_params = request_json
    if str(url).startswith("https://"):
        cur_source_dir = os.path.dirname(__file__)
        cert_path = os.path.join(cur_source_dir, "ssl", "open.fedml.ai_bundle.crt")
        requests.session().verify = cert_path
        response = requests.post(url, json=json_params, verify=True, headers={'Connection': 'close'})
    else:
        response = requests.post(url, json=json_params, headers={'Connection': 'close'})
    status_code = response.json().get("code")
    if status_code == "SUCCESS":
        return True

    return False


def test_login_with_start_run_by_sending_server_agent_msg(args):
    test_run_id = 873
    _, test_edge_id = load_edge_infos()
    start_train_topic = "flserver_agent/{}/start_train".format(str(test_edge_id))
    start_train_msg = {
        "edges": [
            {
                "device_id": "@0x9801a7a55e85.MacOS",
                "os_type": "MacOS",
                "id": test_edge_id
            }
        ],
        "starttime": 1651664950745,
        "url": "http://fedml-server-agent-svc.fedml-aggregator-dev.svc.cluster.local:5001/api/start_run",
        "edgeids": [
            test_edge_id
        ],
        "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6MTA1LCJhY2NvdW50IjoiYWxleC5saWFuZzIiLCJsb2dpblRpbWUiOiIxNjUxNjY0NzM1NDA1IiwiZXhwIjowfQ.vtgrNBhcCVy_d2oI9NeSFcwPxHHSWvtoZS_k-_SLAyU",
        "urls": [],
        "userids": [
            "105"
        ],
        "name": "eye_rain",
        "runId": test_run_id,
        "id": test_run_id,
        "projectid": "121",
        "run_config": {
            "configName": "test-new-open",
            "userId": 105,
            "model_config": {},
            "packages_config": {
                "server": "server-package.zip",
                "linuxClient": "client-package.zip",
                "serverUrl": "https://fedml.s3.us-west-1.amazonaws.com/1651664769220server-package.zip",
                "linuxClientUrl": "https://fedml.s3.us-west-1.amazonaws.com/1651664778846client-package.zip",
                "androidClient": "",
                "androidClientUrl": "",
                "androidClientVersion": "0"
            },
            "data_config": {
                "privateLocalData": "",
                "syntheticData": "",
                "syntheticDataUrl": ""
            },
            "parameters": {
                "model_args": {
                    "model_file_cache_folder": "./model_file_cache",
                    "model": "lr",
                    "global_model_file_path": "./model_file_cache/global_model.pt"
                },
                "device_args": {
                    "worker_num": 2,
                    "using_gpu": False,
                    "gpu_mapping_key": "mapping_default",
                    "gpu_mapping_file": "config/gpu_mapping.yaml"
                },
                "comm_args": {
                    "s3_config_path": "config/s3_config.yaml",
                    "backend": "MQTT_S3",
                    "mqtt_config_path": "config/mqtt_config.yaml"
                },
                "train_args": {
                    "batch_size": 10,
                    "weight_decay": 0.001,
                    "client_num_per_round": 2,
                    "client_num_in_total": 2,
                    "comm_round": 50,
                    "client_optimizer": "sgd",
                    "client_id_list": "[1, 2]",
                    "epochs": 1,
                    "learning_rate": 0.03,
                    "federated_optimizer": "FedAvg"
                },
                "environment_args": {
                    "bootstrap": "config/bootstrap.sh"
                },
                "validation_args": {
                    "frequency_of_the_test": 1
                },
                "common_args": {
                    "random_seed": 0,
                    "training_type": "cross_silo",
                    "using_mlops": False
                },
                "data_args": {
                    "partition_method": "hetero",
                    "partition_alpha": 0.5,
                    "dataset": "mnist",
                    "data_cache_dir": "../../../../data/mnist"
                },
                "tracking_args": {
                    "wandb_project": "fedml",
                    "wandb_name": "fedml_torch_fedavg_mnist_lr",
                    "wandb_key": "ee0b5f53d949c84cee7decbe7a629e63fb2f8408",
                    "enable_wandb": False,
                    "log_file_dir": "./log"
                }
            }
        },
        "timestamp": "1651664950759"
    }

    send_request_to_server_agent(args, start_train_msg)

    time.sleep(3)

    test_is_ok(test_run_id, test_edge_id,
               test_login_with_start_run_by_sending_server_agent_msg, "Connection is ready!")


def test_login_with_stop_run_by_sending_server_agent_msg(args):
    test_run_id = 873
    _, test_edge_id = load_edge_infos()
    stop_train_msg = {
        "edgeids": [
            test_edge_id
        ],
        "runId": test_run_id,
    }

    send_request_to_server_agent(args, stop_train_msg)

    time.sleep(3)

    test_is_ok(test_run_id, test_edge_id,
               test_login_with_stop_run_by_sending_server_agent_msg, "Stop run successfully.")


if __name__ == "__main__":
    try:
        home_dir = expanduser("~")
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--edge_id", "-e", type=int,
                            default=105)
        parser.add_argument("--config_version", "-v", type=str,
                            default="release")
        parser.add_argument("--log_dir", "-l", type=str,
                            default=os.path.join(home_dir, "fedml-client", "fedml", "logs"))
        args = parser.parse_args()
    except Exception as e:
        print("Exception when parsing arguments: {}".format(traceback.format_exc()))
        pass

    test_login_with_start_run_by_sending_client_agent_msg(args)

    #test_login_with_start_run_by_sending_server_agent_msg(args)
    time.sleep(10)
    test_login_with_stop_run_by_sending_client_agent_msg(args)

