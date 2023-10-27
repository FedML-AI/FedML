import argparse
import logging
import os
import platform
import time
import traceback

import click
import fedml
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.master.server_runner import FedMLServerRunner
from fedml.computing.scheduler.master.server_constants import ServerConstants


def __login_as_edge_server_and_agent(args, userid, version, api_key="", use_extra_device_id_suffix=None, role=None):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ServerConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    if hasattr(args, "os_name") and args.os_name is not None and args.os_name != "":
        pass
    else:
        setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", ServerConstants.get_log_file_dir())
    is_from_docker = False
    if hasattr(args, "device_id") and args.device_id is not None and args.device_id != "0":
        setattr(args, "current_device_id", args.device_id)
        is_from_docker = True
    else:
        setattr(args, "current_device_id", FedMLServerRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")

    # Create server runner for communication with the FedML client.
    runner = FedMLServerRunner(args)
    runner.run_as_edge_server_and_agent = True

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
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_1, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("[5] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Judge whether running from fedml docker hub
    is_from_fedml_docker_hub = False
    dock_loc_file = ServerConstants.get_docker_location_file()
    if os.path.exists(dock_loc_file):
        is_from_fedml_docker_hub = True

    # Build unique device id
    if is_from_docker:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".Docker.Edge.Server"
    else:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".Edge.Server"
    setattr(args, "is_from_docker", is_from_docker)

    if is_from_fedml_docker_hub:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".DockerHub.Edge.Server"

    if use_extra_device_id_suffix is not None:
        unique_device_id = args.current_device_id + "@" + args.os_name + use_extra_device_id_suffix

    # Bind account id to FedML® Nexus AI Platform
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
                break
        except Exception as e:
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_2, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            register_try_count += 1
            time.sleep(3)
            continue

    if edge_id <= 0:
        click.echo("")
        click.echo("[6] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return
    setattr(args, "server_id", edge_id)
    runner.args = args
    runner.edge_id = edge_id
    init_logs(args, edge_id)

    # Log arguments and binding results.
    # logging.info("login: unique_device_id = %s" % str(unique_device_id))
    # logging.info("login: server_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    ServerConstants.save_runner_infos(args.current_device_id + "." + args.os_name, edge_id)

    # Setup MQTT connection for communication with the FedML server.
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def __login_as_cloud_agent(args, userid, version):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ServerConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", ServerConstants.get_log_file_dir())
    if hasattr(args, "device_id") and args.device_id is not None and args.device_id != "0":
        setattr(args, "current_device_id", args.device_id)
    else:
        setattr(args, "current_device_id", FedMLServerRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")

    # Create server runner for communication with the FedML client.
    runner = FedMLServerRunner(args)
    runner.run_as_cloud_agent = True

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
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_1, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("[7] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Build unique device id
    if args.current_device_id is not None and len(str(args.current_device_id)) > 0:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".Public.Cloud"

    # Bind account id to FedML® Nexus AI Platform
    register_try_count = 0
    if hasattr(args, "server_agent_id") and args.server_agent_id is not None:
        edge_id = args.server_agent_id
    else:
        edge_id = 0
    while register_try_count < 5:
        try:
            edge_id, user_name, extra_url = runner.bind_account_and_device_id(
                service_config["ml_ops_config"]["EDGE_BINDING_URL"], args.account_id, unique_device_id, args.os_name
            )
            if edge_id > 0:
                runner.edge_id = edge_id
                break
        except Exception as e:
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_2, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            register_try_count += 1
            time.sleep(3)
            continue

    if edge_id <= 0:
        click.echo("")
        click.echo("[8] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return
    setattr(args, "server_id", edge_id)
    runner.args = args
    runner.edge_id = edge_id
    init_logs(args, edge_id)
    logging.info("args {}".format(args))

    # Log arguments and binding results.
    logging.info("login: unique_device_id = %s" % str(unique_device_id))
    logging.info("login: server_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    ServerConstants.save_runner_infos(args.current_device_id + "." + args.os_name, edge_id)

    # Setup MQTT connection for communication with the FedML server.
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def __login_as_cloud_server(args, userid, version):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ServerConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", ServerConstants.get_log_file_dir())
    if hasattr(args, "device_id") and args.device_id is not None and args.device_id != "0":
        setattr(args, "current_device_id", args.device_id)
    else:
        setattr(args, "current_device_id", FedMLServerRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")

    # Create server runner for communication with the FedML client.
    runner = FedMLServerRunner(args)
    runner.run_as_cloud_server = True

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
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_1, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("[9] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Build unique device id
    if hasattr(args, "device_id") and args.device_id is not None and args.device_id != "0":
        unique_device_id = args.current_device_id
    else:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".Public.Server"

    # Bind account id to FedML® Nexus AI Platform
    register_try_count = 0
    edge_id = 0
    while register_try_count < 5:
        try:
            edge_id, user_name, extra_url = runner.bind_account_and_device_id(
                service_config["ml_ops_config"]["EDGE_BINDING_URL"], args.account_id, unique_device_id, args.os_name
            )
            if edge_id > 0:
                runner.edge_id = edge_id
                break
        except Exception as e:
            click.echo("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_2, traceback.format_exc()))
            click.echo(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
            register_try_count += 1
            time.sleep(3)
            continue

    if edge_id <= 0:
        click.echo("")
        click.echo("[10] Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return
    setattr(args, "server_id", edge_id)
    runner.args = args
    runner.edge_id = edge_id
    init_logs(args, edge_id)

    # Log arguments and binding results.
    logging.info("login: unique_device_id = %s" % str(unique_device_id))
    logging.info("login: server_id = %s" % str(edge_id))
    ServerConstants.save_runner_infos(args.current_device_id + "." + args.os_name, edge_id)

    # Echo results
    print("\n\nCongratulations, your device is connected to the FedML MLOps platform successfully!")
    print(
        "Your unique device ID is "
        + str(unique_device_id)
        + "\n"
    )

    # Setup MQTT connection for communication with the FedML server.
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def init_logs(args, edge_id):
    # Init runtime logs
    args.log_file_dir = ServerConstants.get_log_file_dir()
    args.run_id = 0
    args.role = "server"
    args.edge_id = edge_id
    setattr(args, "using_mlops", True)
    setattr(args, "server_agent_id", edge_id)
    # MLOpsRuntimeLog.get_instance(args).init_logs(show_stdout_log=True)


def login(args):
    if args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_LOCAL_INDEX]:
        __login_as_edge_server_and_agent(args, args.user, args.version, api_key=args.api_key)
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_CLOUD_AGENT_INDEX]:
        __login_as_cloud_agent(args, args.user, args.version)
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_CLOUD_SERVER_INDEX]:
        __login_as_cloud_server(args, args.user, args.version)
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_GPU_MASTER_SERVER_INDEX]:
        __login_as_edge_server_and_agent(args, args.user, args.version, api_key=args.api_key,
                                         use_extra_device_id_suffix=".Edge.GPU.MasterServer", role=args.role)


def logout():
    ServerConstants.cleanup_run_process(None)
    sys_utils.cleanup_all_fedml_server_api_processes()


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="local")
    parser.add_argument("--runner_cmd", "-rc", type=str, default="{}")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")

    args = parser.parse_args()
    args.user = args.user
    if args.api_key == "":
        args.api_key = args.user

    fedml.set_env_version(args.version)
    if args.type == 'login':
        login(args)
    else:
        logout()
