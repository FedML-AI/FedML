import os
from os.path import expanduser

import click

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.master.docker_login import logs_with_server_docker_mode
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.slave.docker_login import logs_with_docker_mode


def log(client, server, docker, docker_rank):
    is_client = client
    is_server = server
    if client is None and server is None:
        is_client = True

    is_docker = docker
    if docker is None:
        is_docker = False

    if is_client:
        if is_docker:
            logs_with_docker_mode(docker_rank)
            return
        _display_client_logs()

    if is_server:
        if is_docker:
            logs_with_server_docker_mode(docker_rank)
            return
        _display_server_logs()


def _display_client_logs():
    run_id, edge_id = sys_utils.get_running_info(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                                 ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedml-run-{}-edge-{}.log".format(
        home_dir, ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, str(run_id), str(edge_id)
    )

    if os.path.exists(log_file):
        with open(log_file) as file_handle:
            log_lines = file_handle.readlines()
        for log_line in log_lines:
            click.echo(log_line, nl=False)
    print("\nconsole log file path = {}".format(log_file))


def _display_server_logs():
    run_id, edge_id = sys_utils.get_running_info(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                                 ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedml-run-{}-edge-{}.log".format(
        home_dir, ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, str(run_id), str(edge_id)
    )
    if os.path.exists(log_file):
        with open(log_file) as file_handle:
            log_lines = file_handle.readlines()
        for log_line in log_lines:
            click.echo(log_line)
    print("\nconsole log file path = {}".format(log_file))

