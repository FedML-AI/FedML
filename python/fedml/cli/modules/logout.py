import click

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.master.docker_login import logout_with_server_docker_mode
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.master.server_login import logout as server_logout
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.slave.client_login import logout as client_logout
from fedml.computing.scheduler.slave.docker_login import logout_with_docker_mode


@click.command("logout", help="Logout from MLOps platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--client", "-c", default=None, is_flag=True, help="logout from the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="logout from the FedML server.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="logout from docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default=None, help="docker client rank index (from 1 to n).",
)
def mlops_logout(client, server, docker, docker_rank):
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
    print("\nlogout successfully!\n")
