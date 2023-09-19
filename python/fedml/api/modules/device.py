import os

import click

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


def bind(
        userid, version, client, server,
        api_key, local_server, role, runner_cmd, device_id, os_name,
        docker, docker_rank
):
    print("\n Welcome to FedML.ai! \n Start to login the current device to the FedML® Launch platform "
          "(https://open.fedml.ai)...\n")
    if userid is None or len(userid) <= 0:
        click.echo("Please specify your account id or API key, usage: fedml login $your_account_id_or_api_key")
        return
    account_id = userid[0]
    platform_url = "open.fedml.ai"
    if version != "release":
        platform_url = "open-{}.fedml.ai".format(version)

    # Check user id.
    if userid == "":
        click.echo(
            "Please provide your account id or API key in the FedML® Launch platform ({}).".format(
                platform_url
            )
        )
        return

    # Set client as default entity.
    is_client = client
    is_server = server
    if client is None and server is None:
        is_client = True

    # Check if -c, -s, -l are mutually exclusive
    role_count = (1 if is_client else 0) + (1 if is_server else 0)
    if role_count > 1:
        click.echo("Please make sure you don't specify multiple options between -c, -s.")
        return

    # Set the role
    if is_client:
        default_role = ClientConstants.login_index_role_map[ClientConstants.LOGIN_MODE_CLIEN_INDEX]
        role_index = ClientConstants.login_role_index_map.get(role, ClientConstants.LOGIN_MODE_CLIEN_INDEX)
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
            login_with_docker_mode(account_id, version, docker_rank)
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
                "-ls",
                local_server,
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
                "-ls",
                local_server,
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


def unbind(client, server, docker, docker_rank):
    is_client = client
    is_server = server
    if client is None and server is None:
        is_client = True

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
