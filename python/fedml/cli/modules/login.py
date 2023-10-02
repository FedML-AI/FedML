import click

import fedml.api


@click.command("login", help="Bind to the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.argument("userid", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="bind to which version of FedML® Launch platform. It should be dev, test or release",
)
@click.option(
    "--client", "-c", default=None, is_flag=True, help="bind as the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="bind as the FedML server.",
)
@click.option(
    "--api_key", "-k", type=str, default="", help="user api key.",
)
@click.option(
    "--role",
    "-r",
    type=str,
    default="",
    help="run as the role (options: client, edge_simulator, gpu_supplier, "
         "edge_server, cloud_agent, cloud_server, gpu_master_server.",
)
@click.option(
    "--runner_cmd",
    "-rc",
    type=str,
    default="{}",
    help="runner commands (options: request json for start run, stop run).",
)
@click.option(
    "--device_id", "-id", type=str, default="0", help="device id.",
)
@click.option(
    "--os_name", "-os", type=str, default="", help="os name.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="bind with docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default="1", help="docker client rank index (from 1 to n).",
)
def fedml_login(userid, version, client, server,
                api_key, role, runner_cmd, device_id, os_name,
                docker, docker_rank):
    fedml.set_env_version(version)
    fedml.api.login(userid, client, server,
                    api_key, role, runner_cmd, device_id, os_name,
                    docker, docker_rank)
