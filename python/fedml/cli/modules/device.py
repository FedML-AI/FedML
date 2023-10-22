import click

import fedml.api


@click.group("device")
@click.help_option("--help", "-h")
def fedml_device():
    """
    Bind/unbind devices to the FedML® Nexus AI Platform
    """
    pass


@fedml_device.command("bind", help="Bind to the FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("api_key", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="Bind to which version of FedML® Nexus AI Platform. It should be dev, test or release.",
)
@click.option(
    "--computing", "-c", default=None, is_flag=True,
    help="Bind as the FedML general computing device, which is the default login option."
         "You can not specify the option -c and -s simultaneously.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True,
    help="Bind as the FedML on-premise federated-learning server."
         "You can not specify the option -c and -s simultaneously.",
)
@click.option(
    "--supplier", "-p", default=None, is_flag=True,
    help="Bind as the FedML supplier computing device which will connect to a pay-in account."
         "You can specify the option -p and -c simultaneously, but you can not specify -p and -s simultaneously.",
)
def fedml_device_bind(api_key, version, computing, server, supplier):
    fedml.set_env_version(version)

    api_key = api_key[0]
    
    fedml.api.device_bind(api_key, computing, server, supplier)


@fedml_device.command("unbind", help="Unbind from the FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="Unbind from which version of FedML® Launch platform. It should be dev, test or release",
)
@click.option(
    "--computing", "-c", default=None, is_flag=True, help="Unbind from the FedML general computing device.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="Unbind from the FedML on-premise federated-learning server.",
)
def fedml_device_unbind(version, computing, server):
    fedml.set_env_version(version)

    fedml.api.logout(computing, server)


@fedml_device.command("gpu-type", help="Show gpu resource type (e.g., NVIDIA A100, etc.)")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="show resource type at which version of FedML® Nexus AI Platform. It should be dev, test or release",
)
def resource_type(version):
    fedml.set_env_version(version)
    fedml.api.resource_type()