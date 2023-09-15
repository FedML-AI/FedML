
import argparse
import json
import logging
import os
import platform
import subprocess
import time
import traceback

import click
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.slave.client_runner import FedMLClientRunner
from fedml.computing.scheduler.slave.client_constants import ClientConstants


def init_logs(args, edge_id):
    # Init runtime logs
    args.log_file_dir = ClientConstants.get_log_file_dir()
    args.run_id = 0
    args.role = "client"
    client_ids = list()
    client_ids.append(edge_id)
    args.client_id_list = json.dumps(client_ids)
    setattr(args, "using_mlops", True)
    # MLOpsRuntimeLog.get_instance(args).init_logs(show_stdout_log=True)
    # logging.info("client ids:{}".format(args.client_id_list))


def __login_as_client(args, userid, version, api_key="", use_extra_device_id_suffix=None, role="client"):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ClientConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    if hasattr(args, "os_name") and args.os_name is not None and args.os_name != "":
        pass
    else:
        setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", ClientConstants.get_log_file_dir())
    is_from_docker = False
    if hasattr(args, "device_id") and args.device_id is not None and args.device_id != "0":
        setattr(args, "current_device_id", args.device_id)
        is_from_docker = True
    else:
        setattr(args, "current_device_id", FedMLClientRunner.get_device_id())
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
            # click.echo("service_config = {}".format(service_config))
            log_server_url = mlops_config.get("LOG_SERVER_URL", None)
            if log_server_url is not None:
                setattr(args, "log_server_url", log_server_url)
                setattr(runner.args, "log_server_url", log_server_url)
            break
        except Exception as e:
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_1, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("[1] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Judge whether running from fedml docker hub
    is_from_fedml_docker_hub = False
    dock_loc_file = ClientConstants.get_docker_location_file()
    if os.path.exists(dock_loc_file):
        is_from_fedml_docker_hub = True

    # Build unique device id
    if is_from_docker:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".Docker.Edge.Device"
    else:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".Edge.Device"
    if is_from_fedml_docker_hub:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".DockerHub.Edge.Device"

    if use_extra_device_id_suffix is not None:
        unique_device_id = args.current_device_id + "@" + args.os_name + use_extra_device_id_suffix

    # Bind account id to the MLOps platform.
    register_try_count = 0
    edge_id = 0
    while register_try_count < 5:
        try:
            edge_id, user_name, extra_url = runner.bind_account_and_device_id(
                service_config["ml_ops_config"]["EDGE_BINDING_URL"], args.account_id, unique_device_id, args.os_name,
                api_key=api_key, role=role
            )
            if edge_id > 0:
                runner.edge_id = edge_id
                runner.edge_user_name = user_name
                runner.edge_extra_url = extra_url
                break
        except Exception as e:
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_2, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            register_try_count += 1
            time.sleep(3)
            continue

    if edge_id <= 0:
        click.echo("")
        click.echo("[2] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Init runtime logs
    setattr(args, "client_id", edge_id)
    setattr(args, "is_from_docker", is_from_docker)
    runner.args = args
    init_logs(args, edge_id)
    # logging.info("args {}".format(args))

    # Log arguments and binding results.
    # logging.info("login: unique_device_id = %s" % str(unique_device_id))
    # logging.info("login: edge_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    ClientConstants.save_runner_infos(args.current_device_id + "." + args.os_name, edge_id, run_id=0)

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
    setattr(args, "current_device_id", FedMLClientRunner.get_device_id())
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
        click.echo("[3] Oops, you failed to login the FedML MLOps platform.")
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
            edge_id, _, _ = runner.bind_account_and_device_id(
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
        click.echo("[4] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return False, edge_id, args

    # Init runtime logs
    setattr(args, "client_id", edge_id)
    runner.args = args
    #init_logs(args, edge_id)
    logging.info("args {}".format(args))

    # Log arguments and binding results.
    logging.info("login: unique_device_id = %s" % str(unique_device_id))
    logging.info("login: edge_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id

    if mqtt_connection:
        ClientConstants.save_runner_infos(args.device_id + "." + args.os_name, edge_id, run_id=0)

        # Setup MQTT connection for communication with the FedML server.
        try:
            runner.setup_agent_mqtt_connection(service_config)
        except Exception as e:
            pass

        # Open simulator daemon process to process run status.
        simulator_daemon_cmd = os.path.join(os.path.dirname(__file__), "simulator_daemon.py")
        simulator_daemon_process = sys_utils.run_subprocess_open(
            [
                sys_utils.get_python_program(),
                simulator_daemon_cmd,
                "-t",
                "login",
                "-u",
                str(args.user),
                "-v",
                args.version,
                "-ls",
                args.local_server,
                "-r",
                args.role,
                "-id",
                args.device_id,
                "-os",
                args.os_name,
                "-rk",
                "1",
                "-lfd",
                args.log_file_dir,
                "-cf",
                args.config_version,
                "-ci",
                str(edge_id)
            ]
        ).pid

        # Start mqtt looper
        runner.start_agent_mqtt_loop()

    return True, edge_id, args


def login(args):
    if args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_CLIEN_INDEX]:
        __login_as_client(args, args.user, args.version, api_key=args.api_key)
    elif args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_GPU_SUPPLIER_INDEX]:
        if args.no_gpu_check == 0:
            gpu_count, _ = sys_utils.get_gpu_count_vendor()
            if gpu_count <= 0:
                click.echo("We can't find any gpu device on your machine. \n"
                           "With the gpu_supplier(-g) option, you need to check if your machine "
                           "has nvidia GPUs and installs CUDA related drivers.")
                return
        __login_as_client(args, args.user, args.version, api_key=args.api_key,
                          use_extra_device_id_suffix=".Edge.GPU.Supplier", role=args.role)
    elif args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_EDGE_SIMULATOR_INDEX]:
        __login_as_simulator(args, args.user, args.version)


def logout():
    ClientConstants.cleanup_run_process(None)
    sys_utils.cleanup_all_fedml_client_api_processes()


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="client")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")
    parser.add_argument("--no_gpu_check", "-ngc", type=int, default=1)
    args = parser.parse_args()
    
    args.user = args.user
    if args.type == 'login':
        login(args)
    else:
        logout()


