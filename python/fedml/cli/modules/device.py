import click

import fedml.api


@click.group("device")
@click.help_option("--help", "-h")
def fedml_device():
    """
    Manage devices on the FedML® Launch platform (open.fedml.ai).
    """
    pass


@fedml_device.command("bind", help="Bind to the FedML Platform")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="bind to which version of FedML® Launch platform. It should be dev, test or release",
)
@click.argument("api_key", nargs=-1)
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
def fedml_device_bind(
        version, api_key, client, server,
        role, runner_cmd, device_id, os_name,
        docker, docker_rank, infer_host,
        redis_addr, redis_port, redis_password):
    fedml.set_env_version(version)
    
    # the backend view userid and api_key the same as apiKey.
    userid = api_key[0]
    api_key = api_key[0]
    
    fedml.api.device_bind(userid, client, server,
                          api_key, role, runner_cmd, device_id, os_name,
                          docker, docker_rank, infer_host,
                          redis_addr, redis_port, redis_password)


@fedml_device.command("unbind", help="Logout from the FedML AI Platform")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="unbind to which version of FedML® Launch platform. It should be dev, test or release",
)
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
def fedml_device_unbind(version, client, server, docker, docker_rank):
    fedml.set_env_version(version)
    fedml.api.device_unbind(client, server, docker, docker_rank)


@fedml_device.command("gpu-type", help="Show gpu resource type (e.g., NVIDIA A100, etc.)")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="show resource type at which version of FedML® Launch platform. It should be dev, test or release",
)
def resource_type(version):
    fedml.set_env_version(version)
    fedml.api.resource_type()