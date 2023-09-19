import click

import fedml.api


@click.command("logout", help="unbind from the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--client", "-c", default=None, is_flag=True, help="unbind from the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="unbind from the FedML server.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="unbind from docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default=None, help="docker client rank index (from 1 to n).",
)
def fedml_logout(client, server, docker, docker_rank):
    fedml.api.logout(client, server, docker, docker_rank)
