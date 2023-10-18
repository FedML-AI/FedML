import os

import click

import fedml.api


@click.command("login", help="Login the FedML Platform")
@click.help_option("--help", "-h")
@click.argument("api_key", nargs=-1)
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
@click.option(
    "--infer_host", "-ih", default="127.0.0.1", help="inference host address.",
)
@click.option(
    "--redis_addr", "-ra", default="local", help="inference redis address.",
)
@click.option(
    "--redis_port", "-rp", default="6379", help="inference redis port.",
)
@click.option(
    "--redis_password", "-rpw", default="fedml_default", help="inference redis password.",
)
def fedml_login(api_key, version, client, server,
                role, runner_cmd, device_id, os_name,
                docker, docker_rank, infer_host,
                redis_addr, redis_port, redis_password):
    fedml.set_env_version(version)
    
    # the backend view userid and api_key the same as apiKey.
    userid = api_key[0]
    api_key = api_key[0]
    
    fedml.api.login(userid, client, server,
                    api_key, role, runner_cmd, device_id, os_name,
                    docker, docker_rank, infer_host,
                    redis_addr, redis_port, redis_password)
