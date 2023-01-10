import argparse
import logging
import os
import platform
import time

import click
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.cli.model_deployment.device_server_runner import FedMLServerRunner
from fedml.cli.model_deployment.device_server_constants import ServerConstants


def __login_as_edge_server_and_agent(args, userid, version):
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
    dock_loc_file = ServerConstants.get_docker_location_file()
    if os.path.exists(dock_loc_file):
        is_from_fedml_docker_hub = True

    role_str = ""
    if args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_ON_PREMISE_MASTER_INDEX]:
        role_str = "OnPremise"
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_FEDML_CLOUD_MASTER_INDEX]:
        role_str = "FedMLCloud"
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_PUBLIC_CLOUD_MASTER_INDEX]:
        role_str = "PublicCloud"
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_INFERENCE_INSTANCE_INDEX]:
        role_str = "InferenceInstance"

    # Build unique device id
    is_from_k8s = ServerConstants.is_running_on_k8s()
    if is_from_k8s:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.K8S." + role_str + ".Master.Device"
    elif is_from_docker:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.Docker." + role_str + ".Master.Device"
    else:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA." + role_str + ".Master.Device"

    if is_from_fedml_docker_hub:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.DockerHub." + role_str + ".Master.Device"

    # Bind account id to the ModelOps platform.
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
        click.echo("Oops, you failed to login the FedML ModelOps platform.")
        click.echo("Please check whether your network is normal!")
        return
    setattr(args, "server_id", edge_id)
    runner.args = args
    runner.edge_id = edge_id
    runner.infer_host = args.infer_host
    runner.redis_addr = args.redis_addr
    runner.redis_port = args.redis_port
    runner.redis_password = args.redis_password
    init_logs(edge_id)

    # Log arguments and binding results.
    logging.info("login: unique_device_id = %s" % str(unique_device_id))
    logging.info("login: server_id = %s" % str(edge_id))
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
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("Oops, you failed to login the FedML ModelOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Build unique device id
    if args.current_device_id is not None and len(str(args.current_device_id)) > 0:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.Public.Cloud.Device"

    # Bind account id to the ModelOps platform.
    register_try_count = 0
    if hasattr(args, "server_agent_id") and args.server_agent_id is not None:
        edge_id = args.server_agent_id
    else:
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
        click.echo("Oops, you failed to login the FedML ModelOps platform.")
        click.echo("Please check whether your network is normal!")
        return
    setattr(args, "server_id", edge_id)
    runner.args = args
    runner.edge_id = edge_id
    runner.infer_host = args.infer_host
    runner.redis_addr = args.redis_addr
    runner.redis_port = args.redis_port
    runner.redis_password = args.redis_password
    init_logs(edge_id)
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
            config_try_count += 1
            time.sleep(3)
            continue

    if config_try_count >= 5:
        click.echo("")
        click.echo("Oops, you failed to login the FedML ModelOps platform.")
        click.echo("Please check whether your network is normal!")
        return

    # Build unique device id
    if hasattr(args, "device_id") and args.device_id is not None and args.device_id != "0":
        unique_device_id = args.current_device_id
    else:
        unique_device_id = args.current_device_id + "@" + args.os_name + ".MDA.Public.Cloud.DeviceInstance"

    # Bind account id to the ModelOps platform.
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
        click.echo("Oops, you failed to login the FedML ModelOps platform.")
        click.echo("Please check whether your network is normal!")
        return
    setattr(args, "server_id", edge_id)
    runner.args = args
    runner.edge_id = edge_id
    runner.infer_host = args.infer_host
    runner.redis_addr = args.redis_addr
    runner.redis_port = args.redis_port
    runner.redis_password = args.redis_password
    init_logs(edge_id)

    # Log arguments and binding results.
    logging.info("login: unique_device_id = %s" % str(unique_device_id))
    logging.info("login: server_id = %s" % str(edge_id))
    ServerConstants.save_runner_infos(args.current_device_id + "." + args.os_name, edge_id)

    # Echo results
    logging.info("Congratulations, you have logged into the FedML ModelOps platform successfully!")
    logging.info("Your server unique device id is " + str(unique_device_id))

    # Start the FedML server
    runner.callback_start_train(payload=args.runner_cmd)


def init_logs(edge_id):
    # Init runtime logs
    args.log_file_dir = ServerConstants.get_log_file_dir()
    args.run_id = 0
    args.rank = 0
    args.edge_id = edge_id
    setattr(args, "using_mlops", True)
    setattr(args, "server_agent_id", edge_id)
    MLOpsRuntimeLog.get_instance(args).init_logs(show_stdout_log=True)


def login(args):
    # Install redis server
    if not hasattr(args, "redis_addr") or args.redis_addr is None or args.redis_addr == "local":
        sys_name = platform.system()
        if sys_name == "Linux":
            os.system("sudo rm -f /usr/share/keyrings/redis-archive-keyring.gpg;curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg")
            os.system("echo \"deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/redis.list")
            os.system("sudo apt-get update")
            os.system("sudo service redis-server stop;sudo apt-get install -y redis && nohup redis-server&")

    if args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_ON_PREMISE_MASTER_INDEX]:
        __login_as_edge_server_and_agent(args, args.user, args.version)
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_FEDML_CLOUD_MASTER_INDEX]:
        __login_as_cloud_agent(args, args.user, args.version)
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_PUBLIC_CLOUD_MASTER_INDEX]:
        __login_as_cloud_agent(args, args.user, args.version)
    elif args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_INFERENCE_INSTANCE_INDEX]:
        __login_as_cloud_server(args, args.user, args.version)


def logout():
    ServerConstants.cleanup_run_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to ModelOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at ModelOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="md.on_premise_device.master")
    parser.add_argument("--runner_cmd", "-rc", type=str, default="{}")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--infer_host", "-ih", type=str, default="127.0.0.1")
    parser.add_argument("--redis_addr", "-ra", type=str, default="local")
    parser.add_argument("--redis_port", "-rp", type=str, default="6379")
    parser.add_argument("--redis_password", "-rpw", type=str, default="fedml_default")

    args = parser.parse_args()
    args.user = args.user
    if args.type == 'login':
        login(args)
    else:
        logout()
