import click

from fedml.cli.modules import utils
from fedml.computing.scheduler.master.docker_login import logs_with_server_docker_mode
from fedml.computing.scheduler.slave.docker_login import logs_with_docker_mode


@click.command("logs", help="Display fedml logs.")
@click.help_option("--help", "-h")
@click.option(
    "--client", "-c", default=None, is_flag=True, help="Display client logs.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="Display server logs.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="Display client docker logs.",
)
@click.option(
    "--docker-rank", "-dr", default="1", help="docker client rank index (from 1 to n).",
)
def fedml_logs(client, server, docker, docker_rank):
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
        utils.display_client_logs()

    if is_server:
        if is_docker:
            logs_with_server_docker_mode(docker_rank)
            return
        utils.display_server_logs()
