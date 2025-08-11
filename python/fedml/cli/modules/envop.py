import click
import fedml.api
from fedml.cli.modules.utils import OrderedGroup
from fedml.computing.scheduler.env.collect_env import collect_env


@click.group("envop", cls=OrderedGroup)
@click.help_option("--help", "-h")
def fedml_env():
    """
     FedML Env CLI will help you check your environment info such as versions, hardware, and networking,
     and also you can use it to set your environment info for accessing the remote server, for example,
     the remote server url and port for on-premise mode.
    """
    pass


@fedml_env.command("get", help="Get environment info such as versions, hardware, and networking")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="support values: local, dev, test, release",
)
def fedml_env_get(version):
    fedml.set_env_version(version)
    collect_env()


@fedml_env.command("set", help="Set environment info such as versions, hardware, and networking")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="local",
    help="support values: local, dev, test, release",
)
@click.option(
    "--local_on_premise_platform",
    "-lp",
    type=str,
    default="127.0.0.1",
    help="The IP address for local on-premise Nexus AI Platform.",
)
@click.option(
    "--local_on_premise_platform_port",
    "-lpp",
    type=int,
    default=80,
    help="The port for local on-premise Nexus AI Platform.",
)
@click.option(
    "--service_provider",
    "-sp",
    type=str,
    default="",
    help="Service provider.",
)
def fedml_env_set(version, local_on_premise_platform, local_on_premise_platform_port, service_provider):
    fedml.set_env_version(version)
    fedml.set_local_on_premise_platform_host(local_on_premise_platform)
    fedml.set_local_on_premise_platform_port(local_on_premise_platform_port)

    if service_provider == "chainopera" or service_provider == "co":
        fedml.set_env_version('local')
        fedml.set_local_on_premise_platform_host('open.chainopera.ai')
        fedml.set_local_on_premise_platform_port(443)

