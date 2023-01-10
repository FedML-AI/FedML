import os
import shutil
import subprocess
from os.path import expanduser

import click

import fedml

from ...cli.model_deployment.device_client_constants import ClientConstants
from ...cli.model_deployment.device_server_constants import ServerConstants
from ...cli.model_deployment.device_client_login import logout as client_logout
from ...cli.env.collect_env import collect_env
from ...cli.model_deployment.device_server_login import logout as server_logout
from ...cli.model_deployment.docker_client_login import login_with_docker_mode
from ...cli.model_deployment.docker_client_login import logout_with_docker_mode
from ...cli.model_deployment.docker_client_login import logs_with_docker_mode
from ...cli.model_deployment.docker_server_login import login_with_server_docker_mode
from ...cli.model_deployment.docker_server_login import logout_with_server_docker_mode
from ...cli.model_deployment.docker_server_login import logs_with_server_docker_mode
from ...cli.comm_utils import sys_utils


def login_as_model_device_agent(
    userid, cloud, on_premise, master, infer_host, version, local_server,
    runner_cmd, device_id, os_name, docker, docker_rank, redis_addr, redis_port, redis_password
):
    account_id = userid[0]
    platform_url = "open.fedml.ai"
    if version != "release":
        platform_url = "open-{}.fedml.ai".format(version)

    # Check user id.
    if userid == "":
        click.echo(
            "Please provide your account id in the MLOps platform ({}).".format(
                platform_url
            )
        )
        return

    # click.echo("client {}, server {}".format(client, server))
    # Set client as default entity.
    is_cloud = cloud
    is_on_premise = on_premise
    is_master = master
    if cloud is None and on_premise is None:
        is_on_premise = True
    if master is None:
        is_master = False
    if is_cloud and is_on_premise:
        is_cloud = False

    # Check docker mode.
    is_docker = docker
    if docker is None:
        is_docker = False

    # click.echo("login as client: {}, as server: {}".format(is_client, is_server))
    role = None
    if not is_master:
        if is_docker:
            login_with_docker_mode(account_id, version, docker_rank)
            return
        pip_source_dir = os.path.dirname(__file__)
        login_cmd = os.path.join(pip_source_dir, "device_client_daemon.py")
        client_logout()
        sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_client_learning_processes()
        sys_utils.cleanup_all_fedml_client_login_processes("device_client_login.py")

        if is_on_premise is True:
            role = ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_ON_PREMISE_INDEX]

        if is_cloud is True:
            role = ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_FEDML_CLOUD_INDEX]

        login_pid = subprocess.Popen(
            [
                sys_utils.get_python_program(),
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
                "-ih",
                infer_host
            ]
        ).pid
        sys_utils.save_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                     ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)

    else:
        if is_on_premise is True:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_ON_PREMISE_MASTER_INDEX]

        if is_cloud is True:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_FEDML_CLOUD_MASTER_INDEX]

        if is_docker:
            login_with_server_docker_mode(account_id, version, docker_rank)
            return

        pip_source_dir = os.path.dirname(__file__)
        login_cmd = os.path.join(pip_source_dir, "device_server_daemon.py")
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("device_server_login.py")
        login_pid = subprocess.Popen(
            [
                sys_utils.get_python_program(),
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
                "-ih",
                infer_host,
                "-ra",
                redis_addr,
                "-rp",
                redis_port,
                "-rpw",
                redis_password
            ]
        ).pid
        sys_utils.save_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                     ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)


def logout_from_model_ops(slave, master, docker, docker_rank):
    is_client = slave
    is_server = master
    if slave is None and master is None:
        is_client = True

    is_docker = docker
    if docker is None:
        is_docker = False

    if is_client is True:
        if is_docker:
            logout_with_docker_mode(docker_rank)
            return
        client_logout()
        sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_client_learning_processes()
        sys_utils.cleanup_all_fedml_client_login_processes("device_client_login.py")

    if is_server is True:
        if is_docker:
            logout_with_server_docker_mode(docker_rank)
            return
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("device_server_login.py")


if __name__ == "__main__":
    pass
