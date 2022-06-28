import argparse
import json
import logging
import os
import platform
import time

import click
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.cli.edge_deployment.client_runner import FedMLClientRunner
from fedml.cli.edge_deployment.client_runner import LOCAL_HOME_RUNNER_DIR_NAME as CLIENT_RUNNER_HOME_DIR
from fedml.cli.edge_deployment.client_runner import LOCAL_RUNNER_INFO_DIR_NAME as CLIENT_RUNNER_INFO_DIR


def init_logs(edge_id):
    # Init runtime logs
    args.log_file_dir = FedMLClientRunner.get_log_file_dir()
    args.run_id = 0
    args.rank = 1
    client_ids = list()
    client_ids.append(edge_id)
    args.client_id_list = json.dumps(client_ids)
    click.echo("client ids:{}".format(args.client_id_list))
    MLOpsRuntimeLog.get_instance(args).init_logs()
    logging.info("client ids:{}".format(args.client_id_list))


def __login(args, userid, version):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", FedMLClientRunner.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", os.path.join(args.current_running_dir, "fedml", "logs"))
    setattr(args, "device_id", FedMLClientRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")
    click.echo(args)

    # Create client runner for communication with the FedML server.
    runner = FedMLClientRunner(args)

    # Fetch configs from the MLOps config server.
    service_config = dict()
    config_try_count = 0
    edge_id = 0
    while config_try_count < 5:
        try:
            mqtt_config, s3_config, mlops_config, docker_config = runner.fetch_configs()
            service_config["mqtt_config"] = mqtt_config
            service_config["s3_config"] = s3_config
            service_config["ml_ops_config"] = mlops_config
            service_config["docker_config"] = docker_config
            runner.agent_config = service_config
            break
        except Exception as e:
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Build unique device id
    if args.device_id is not None and len(str(args.device_id)) > 0:
        unique_device_id = "@" + args.device_id + "." + args.os_name

    # Bind account id to the MLOps platform.
    register_try_count = 0
    edge_id = 0
    while register_try_count < 5:
        try:
            edge_id = runner.bind_account_and_device_id(
                service_config["ml_ops_config"]["EDGE_BINDING_URL"], args.account_id, unique_device_id, args.os_name
            )
            if edge_id > 0:
                runner.edge_id = edge_id
                break
        except Exception as e:
            register_try_count += 1
            time.sleep(3)
            continue

    if edge_id <= 0:
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Init runtime logs
    init_logs(edge_id)

    # Log arguments and binding results.
    click.echo("login: unique_device_id = %s" % str(unique_device_id))
    click.echo("login: edge_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    FedMLClientRunner.save_runner_infos(args.device_id + "." + args.os_name, edge_id, run_id=0)

    # Setup MQTT connection for communication with the FedML server.
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def login(args):
    __login(args, args.user, args.version)


def logout():
    FedMLClientRunner.cleanup_run_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    args = parser.parse_args()
    click.echo(args)
    args.user = args.user
    if args.type == 'login':
        login(args)
    else:
        logout(args)
