
import argparse
import json
import os
import platform
import time
import traceback

import click

from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.model_scheduler.device_client_runner import FedMLClientRunner
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants


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


def __login_as_client(args, userid, version):
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

    # Fetch configs from the ModelOps config server.
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
        click.echo("Oops, you failed to login the FedML ModelOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Judge whether running from fedml docker hub
    is_from_fedml_docker_hub = False
    dock_loc_file = ClientConstants.get_docker_location_file()
    if os.path.exists(dock_loc_file):
        is_from_fedml_docker_hub = True

    role_str = ""
    if args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_ON_PREMISE_INDEX]:
        role_str = "OnPremise"
    elif args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_FEDML_CLOUD_INDEX]:
        role_str = "FedMLCloud"
    elif args.role == ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_PUBLIC_CLOUD_INDEX]:
        role_str = "PublicCloud"

    # Build unique device id
    is_from_k8s = ClientConstants.is_running_on_k8s()
    if is_from_k8s:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.K8S." + role_str + ".Device"
    elif is_from_docker:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.Docker." + role_str + ".Device"
    else:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA." + role_str + ".Device"
    if is_from_fedml_docker_hub:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.DockerHub." + role_str + ".Device"

    # Bind account id to the ModelOps platform.
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
        click.echo("Oops, you failed to login the FedML ModelOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Init runtime logs
    setattr(args, "client_id", edge_id)
    runner.args = args
    init_logs(args, edge_id)
    # logging.info("args {}".format(args))

    # Log arguments and binding results.
    # logging.info("login: unique_device_id = %s" % str(unique_device_id))
    # logging.info("login: edge_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    ClientConstants.save_runner_infos(args.current_device_id + "." + args.os_name, edge_id, run_id=0)

    # Setup MQTT connection for communication with the FedML server.
    runner.infer_host = args.infer_host
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def login(args):
    __login_as_client(args, args.user, args.version)


def logout():
    ClientConstants.cleanup_run_process(None)


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to ModelOps platform")
    parser.add_argument("--user", "-u", type=str, help="account id at ModelOps platform")
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="md.on_premise_device")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--infer_host", "-ih", type=str, default="127.0.0.1")
    args = parser.parse_args()
    args.user = args.user
    if args.type == 'login':
        login(args)
    else:
        logout()
