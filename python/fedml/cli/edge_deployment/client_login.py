import argparse
import json
import logging
import platform
import time

import click
from ...core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from ...cli.edge_deployment.client_runner import FedMLClientRunner
from ...cli.edge_deployment.client_constants import ClientConstants

def init_logs(args, edge_id):
    # Init runtime logs
    args.log_file_dir = ClientConstants.get_log_file_dir()
    args.run_id = 0
    args.rank = 1
    client_ids = list()
    client_ids.append(edge_id)
    args.client_id_list = json.dumps(client_ids)
    setattr(args, "using_mlops", True)
    MLOpsRuntimeLog.get_instance(args).init_logs(show_stdout_log=False)
    logging.info("client ids:{}".format(args.client_id_list))


def __login_as_client(args, userid, version):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ClientConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", ClientConstants.get_log_file_dir())
    setattr(args, "device_id", FedMLClientRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")

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
            log_server_url = mlops_config.get("LOG_SERVER_URL", None)
            if log_server_url is not None:
                setattr(args, "log_server_url", log_server_url)
                setattr(runner.args, "log_server_url", log_server_url)
            break
        except Exception as e:
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Build unique device id
    if args.device_id is not None and len(str(args.device_id)) > 0:
        unique_device_id = args.device_id + "@" + args.os_name + ".Edge.Device"

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
        click.echo("")
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Init runtime logs
    init_logs(args, edge_id)
    logging.info("args {}".format(args))

    # Log arguments and binding results.
    logging.info("login: unique_device_id = %s" % str(unique_device_id))
    logging.info("login: edge_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    ClientConstants.save_runner_infos(args.device_id + "." + args.os_name, edge_id, run_id=0)

    # Setup MQTT connection for communication with the FedML server.
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def __login_as_simulator(args, userid, version, mqtt_connection=True):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ClientConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", ClientConstants.get_log_file_dir())
    setattr(args, "device_id", FedMLClientRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")

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
            log_server_url = mlops_config.get("LOG_SERVER_URL", None)
            if log_server_url is not None:
                setattr(args, "log_server_url", log_server_url)
                setattr(runner.args, "log_server_url", log_server_url)
            break
        except Exception as e:
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return False, edge_id, args

    # Build unique device id
    if args.device_id is not None and len(str(args.device_id)) > 0:
        unique_device_id = args.device_id + "@" + args.os_name + ".Edge.Simulator"

    # Bind account id to the MLOps platform.
    register_try_count = 0
    edge_id = 0
    while register_try_count < 5:
        try:
            edge_id = runner.bind_account_and_device_id(
                service_config["ml_ops_config"]["EDGE_BINDING_URL"], args.account_id,
                unique_device_id, args.os_name, role="simulator"
            )
            if edge_id > 0:
                runner.edge_id = edge_id
                break
        except Exception as e:
            register_try_count += 1
            time.sleep(3)
            continue

    if edge_id <= 0:
        click.echo("")
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return False, edge_id, args

    # Init runtime logs
    #init_logs(args, edge_id)
    logging.info("args {}".format(args))

    # Log arguments and binding results.
    logging.info("login: unique_device_id = %s" % str(unique_device_id))
    logging.info("login: edge_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id

    if mqtt_connection:
        ClientConstants.save_runner_infos(args.device_id + "." + args.os_name, edge_id, run_id=0)

        # Setup MQTT connection for communication with the FedML server.
        runner.setup_agent_mqtt_connection(service_config)

        # Start mqtt looper
        runner.start_agent_mqtt_loop()

    return True, edge_id, args


def login(args):
    if args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_CLIEN_INDEX]:
        __login_as_client(args, args.user, args.version)
    elif args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_EDGE_SIMULATOR_INDEX]:
        __login_as_simulator(args, args.user, args.version)


def logout():
    ClientConstants.cleanup_run_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="client")
    args = parser.parse_args()
    args.user = args.user
    if args.type == 'login':
        login(args)
    else:
        logout(args)
