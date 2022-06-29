import argparse
import os
import platform
import time
import click
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.cli.server_deployment.server_runner import FedMLServerRunner
from fedml.cli.server_deployment.server_runner import LOCAL_HOME_RUNNER_DIR_NAME as SERVER_RUNNER_HOME_DIR
from fedml.cli.server_deployment.server_runner import LOCAL_RUNNER_INFO_DIR_NAME as SERVER_RUNNER_INFO_DIR

LOGIN_MODE_LOCAL_INDEX = 0
LOGIN_MODE_CLOUD_AGENT_INDEX = 1
LOGIN_MODE_CLOUD_SERVER_INDEX = 2
login_role_list = ["edge_server", "cloud_agent", "cloud_server"]


def __login_as_edge_server_and_agent(args, userid, version):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", FedMLServerRunner.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", os.path.join(args.current_running_dir, "fedml", "logs"))
    setattr(args, "device_id", FedMLServerRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")
    click.echo(args)

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
        unique_device_id = args.device_id + "@" + args.os_name + ".Edge.Server"

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
    runner.edge_id = edge_id
    init_logs(edge_id)

    # Log arguments and binding results.
    click.echo("login: unique_device_id = %s" % str(unique_device_id))
    click.echo("login: server_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    FedMLServerRunner.save_runner_infos(args.device_id + "." + args.os_name, edge_id)

    # Setup MQTT connection for communication with the FedML server.
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def __login_as_cloud_agent(args, userid, version):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", FedMLServerRunner.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", os.path.join(args.current_running_dir, "fedml", "logs"))
    setattr(args, "device_id", FedMLServerRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")
    click.echo(args)

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
        unique_device_id = args.device_id + "@" + args.os_name + ".Public.Cloud"

    # Bind account id to the MLOps platform.
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
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
        return
    runner.edge_id = edge_id
    init_logs(edge_id)

    # Log arguments and binding results.
    click.echo("login: unique_device_id = %s" % str(unique_device_id))
    click.echo("login: server_id = %s" % str(edge_id))
    runner.unique_device_id = unique_device_id
    FedMLServerRunner.save_runner_infos(args.device_id + "." + args.os_name, edge_id)

    # Setup MQTT connection for communication with the FedML server.
    runner.setup_agent_mqtt_connection(service_config)

    # Start mqtt looper
    runner.start_agent_mqtt_loop()


def __login_as_cloud_server(args, userid, version):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", FedMLServerRunner.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    setattr(args, "log_file_dir", os.path.join(args.current_running_dir, "fedml", "logs"))
    setattr(args, "device_id", FedMLServerRunner.get_device_id())
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")
    click.echo(args)

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
        unique_device_id = args.device_id + "@" + args.os_name + ".Public.Server"

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
    runner.edge_id = edge_id
    init_logs(edge_id)

    # Log arguments and binding results.
    click.echo("login: unique_device_id = %s" % str(unique_device_id))
    click.echo("login: server_id = %s" % str(edge_id))
    FedMLServerRunner.save_runner_infos(args.device_id + "." + args.os_name, edge_id)

    # Echo results
    click.echo("Congratulations, you have logged into the FedML MLOps platform successfully!")
    click.echo("Your server unique device id is " + str(unique_device_id))

    # Start the FedML server
    runner.callback_start_train(payload=args.runner_cmd)


def init_logs(edge_id):
    # Init runtime logs
    args.log_file_dir = FedMLServerRunner.get_log_file_dir()
    args.run_id = 0
    args.rank = 0
    args.edge_id = edge_id
    MLOpsRuntimeLog.get_instance(args).init_logs()


def login(args):
    if args.role == login_role_list[LOGIN_MODE_LOCAL_INDEX]:
        __login_as_edge_server_and_agent(args, args.user, args.version)
    elif args.role == login_role_list[LOGIN_MODE_CLOUD_AGENT_INDEX]:
        __login_as_cloud_agent(args, args.user, args.version)
    elif args.role == login_role_list[LOGIN_MODE_CLOUD_SERVER_INDEX]:
        __login_as_cloud_server(args, args.user, args.version)


def logout():
    FedMLServerRunner.cleanup_run_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="local")
    parser.add_argument("--runner_cmd", "-rc", type=str, default="{}")
    parser.add_argument("--server_agent_id", "-said", type=str, default="0")

    args = parser.parse_args()
    click.echo(args)
    args.user = args.user
    if args.type == 'login':
        login(args)
    else:
        logout(args)
