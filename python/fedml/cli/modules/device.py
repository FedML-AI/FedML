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
    "--compute_node", "-c", default=None, is_flag=True,
    help="Bind as the general compute node in FEDML Nexus AI compute network. This is enabled by default. "
    "After binding, you can view and manage the device in the FEDML® Nexus AI Platform: https://fedml.ai/compute. "
    "It can be grouped as a cluster and then you can use FEDML®Launch to schedule any job (training, deployment, federated learning) to it. "
    "You can not specify the option -c and -s simultaneously.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True,
    help="Bind as the FedML on-premise parameter server (PS). It can be used for PS-based training paradigms, such as distributed training, cross-cloud training, and federated-learning. "
    "You can not specify the option -c and -s simultaneously for a single environment.",
)
@click.option(
    "--provider", "-p", default=None, is_flag=True,
    help="Bind as the FedML compute node (GPU) provider (supplier). This is used by Nexus AI Platform - Share and Earn: https://fedml.ai/gpu-supplier. You can share your GPUs in this way and earn money. "
    "You can specify the option -p and -c simultaneously (can be used as provider for others as well compute node for your own jobs), but you can not specify -p and -s simultaneously.",
)
def fedml_device_bind(api_key, version, compute_node, server, provider):
    fedml.set_env_version(version)

    api_key = api_key[0]
    
    fedml.api.device_bind(api_key, compute_node, server, provider)


@fedml_device.command("unbind", help="Unbind from the FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="Unbind which backend environment version of FedML® Nexus AI Platform. It should be dev, test, or release.",
)
@click.option(
    "--compute_node", "-c", default=None, is_flag=True, help="Unbind from the FedML general compute node.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="Unbind from the the FedML on-premise parameter server (PS).",
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