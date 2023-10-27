import os

import click
from prettytable import PrettyTable

import fedml
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.master.docker_login import login_with_server_docker_mode
from fedml.computing.scheduler.master.docker_login import logout_with_server_docker_mode
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.master.server_login import logout as server_logout
from fedml.computing.scheduler.model_scheduler import device_login_entry
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.slave.client_login import logout as client_logout
from fedml.computing.scheduler.slave.docker_login import login_with_docker_mode
from fedml.computing.scheduler.slave.docker_login import logout_with_docker_mode
from fedml.computing.scheduler.scheduler_entry.resource_manager import FedMLResourceManager


def bind(
        api_key, computing, server, supplier
):
    userid = api_key
    runner_cmd = "{}"
    device_id = "0"
    os_name = ""
    docker = None
    docker_rank = 1
    infer_host = "127.0.0.1"
    redis_addr = "local"
    redis_port = "6379"
    redis_password = "fedml_default"
    role = ""
    is_client = computing
    is_server = server
    is_supplier = supplier
    if supplier is None:
        is_supplier = False
    if is_server and is_supplier:
        print("You can not specify the option -p and -s simultaneously.")
        return
    if is_supplier:
        role = ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_GPU_SUPPLIER_INDEX]
    elif is_server:
        role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_LOCAL_INDEX]
    elif is_client:
        role = ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_CLIENT_INDEX]

    _bind(
        userid, computing, server,
        api_key, role, runner_cmd, device_id, os_name,
        docker, docker_rank, infer_host,
        redis_addr, redis_port, redis_password
    )


def _bind(
        userid, computing, server,
        api_key, role, runner_cmd, device_id, os_name,
        docker, docker_rank, infer_host,
        redis_addr, redis_port, redis_password
):
    os.environ["FEDML_INFER_HOST"] = infer_host
    os.environ["FEDML_INFER_REDIS_ADDR"] = redis_addr
    os.environ["FEDML_INFER_REDIS_PORT"] = redis_port
    os.environ["FEDML_INFER_REDIS_PASSWORD"] = redis_password

    url = fedml._get_backend_service()
    print("\n Welcome to FedML.ai! \n Start to login the current device to the FedML® Nexus AI Platform\n")
    if api_key is None:
        click.echo("Please specify your API key, usage: fedml login $your_api_key")
        return
    account_id = userid
    # print(f"account_id = {account_id}")
    # print(f"api_key = {api_key}")

    # Set client as default entity.
    is_client = computing
    is_server = server
    if computing is None and server is None:
        is_client = True
        is_server = False

    if is_client and is_server:
        print("You can not specify the option -c and -s simultaneously.")
        return

    # Check if -c, -s, -l are mutually exclusive
    role_count = (1 if is_client else 0) + (1 if is_server else 0)
    if role_count > 1:
        click.echo("Please make sure you don't specify multiple options between -c, -s.")
        return

    # Set the role
    if is_client:
        default_role = ClientConstants.login_index_role_map[ClientConstants.LOGIN_MODE_CLIENT_INDEX]
        role_index = ClientConstants.login_role_index_map.get(role, ClientConstants.LOGIN_MODE_CLIENT_INDEX)
        role = ClientConstants.login_index_role_map.get(role_index, default_role)
    elif is_server:
        default_role = ServerConstants.login_index_role_map[ServerConstants.LOGIN_MODE_LOCAL_INDEX]
        role_index = ServerConstants.login_role_index_map.get(role, ServerConstants.LOGIN_MODE_LOCAL_INDEX)
        role = ServerConstants.login_index_role_map.get(role_index, default_role)

    # Check api key
    user_api_key = api_key
    if api_key is None:
        user_api_key = "NONE"

    # Check docker mode.
    is_docker = docker
    if docker is None:
        is_docker = False

    infer_host = "127.0.0.1"
    redis_addr = "local"
    redis_port = "6379"
    redis_password = "fedml_default"

    if is_client is True:
        if is_docker:
            login_with_docker_mode(account_id, docker_rank)
            return
        pip_source_dir = os.path.dirname(__file__)
        pip_source_dir = os.path.dirname(pip_source_dir)
        pip_source_dir = os.path.dirname(pip_source_dir)
        login_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "slave", "client_daemon.py")

        client_logout()
        sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                        ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_client_learning_processes()
        sys_utils.cleanup_all_fedml_client_login_processes("client_login.py")
        sys_utils.cleanup_all_fedml_client_api_processes(kill_all=True)

        version = fedml.get_env_version()
        login_pid = sys_utils.run_subprocess_open(
            [
                sys_utils.get_python_program(),
                "-W",
                "ignore",
                login_cmd,
                "-t",
                "login",
                "-u",
                str(account_id),
                "-v",
                version,
                "-r",
                role,
                "-id",
                device_id,
                "-os",
                os_name,
                "-k",
                user_api_key,
                "-ngc",
                "1"
            ]
        ).pid
        sys_utils.save_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                     ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)

    if is_server is True:
        if is_docker:
            login_with_server_docker_mode(account_id, version, docker_rank)
            return

        pip_source_dir = os.path.dirname(__file__)
        pip_source_dir = os.path.dirname(pip_source_dir)
        pip_source_dir = os.path.dirname(pip_source_dir)
        login_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "master", "server_daemon.py")
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                        ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("server_login.py")
        sys_utils.cleanup_all_fedml_server_api_processes(kill_all=True)
        
        version = fedml.get_env_version()
        login_pid = sys_utils.run_subprocess_open(
            [
                sys_utils.get_python_program(),
                "-W",
                "ignore",
                login_cmd,
                "-t",
                "login",
                "-u",
                str(account_id),
                "-v",
                version,
                "-r",
                role,
                "-rc",
                runner_cmd,
                "-id",
                device_id,
                "-os",
                os_name,
                "-k",
                user_api_key
            ]
        ).pid
        sys_utils.save_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                     ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)


def unbind(computing, server):
    docker = None
    docker_rank = 1
    is_client = computing
    is_server = server
    if computing is None and server is None:
        is_client = True
        is_server = True

    is_docker = docker
    if docker is None:
        is_docker = False

    if is_client is True:
        if is_docker:
            logout_with_docker_mode(docker_rank)
            return
        sys_utils.cleanup_all_fedml_client_login_processes("client_daemon.py")
        client_logout()
        sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                        ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_client_learning_processes()
        sys_utils.cleanup_all_fedml_client_login_processes("client_login.py")
        sys_utils.cleanup_all_fedml_client_api_processes(kill_all=True)
        sys_utils.cleanup_all_fedml_client_login_processes("client_daemon.py")

        device_login_entry.logout_from_model_ops(True, True, docker, docker_rank)

    if is_server is True:
        if is_docker:
            logout_with_server_docker_mode(docker_rank)
            return
        sys_utils.cleanup_all_fedml_server_login_processes("server_daemon.py")
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                        ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("server_login.py")
        sys_utils.cleanup_all_fedml_server_api_processes(kill_all=True)
        sys_utils.cleanup_all_fedml_server_login_processes("server_daemon.py")

        device_login_entry.logout_from_model_ops(False, True, docker, docker_rank)

    print("\nlogout successfully!\n")


def resource_type():
    resource_type_list = FedMLResourceManager.get_instance().show_resource_type()
    if resource_type_list is not None and len(resource_type_list) > 0:
        click.echo("All available resource type is as follows.")
        resource_table = PrettyTable(['Resource Type', 'GPU Type'])
        for type_item in resource_type_list:
            resource_table.add_row([type_item[0], type_item[1]])
        print(resource_table)
    else:
        click.echo("No available resource type.")