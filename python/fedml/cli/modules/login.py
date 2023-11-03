import click

import fedml.api


@click.command("login", help="Login the FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.argument("api_key", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="Login which backend environment version of FedML® Nexus AI Platform. It should be dev, test, or release.",
)
@click.option(
    "--compute_node", "-c", default=None, is_flag=True,
    help="Login as the general compute node in FEDML Nexus AI compute network. This is enabled by default. "
         "After login, you can view and manage the device in the FEDML® Nexus AI Platform: https://nexus.fedml.ai/compute. "
         "It can be grouped as a cluster and then you can use FEDML®Launch to schedule any job (training, deployment, federated learning) to it. "
         "You can not specify the option -c and -s simultaneously.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True,
    help="Login as the FedML on-premise parameter server (PS). It can be used for PS-based training paradigms, such as distributed training, cross-cloud training, and federated-learning. "
         "You can not specify the option -c and -s simultaneously for a single environment.",
)
@click.option(
    "--provider", "-p", default=None, is_flag=True,
    help="Login as the FedML compute node (GPU) provider (supplier). This is used by Nexus AI Platform - Share and Earn: https://nexus.fedml.ai/gpu-supplier. You can share your GPUs in this way and earn money. "
         "You can specify the option -p and -c simultaneously (can be used as provider for others as well compute node for your own jobs), but you can not specify -p and -s simultaneously.",
)
def fedml_login(api_key, version, compute_node, server, provider):
    fedml.set_env_version(version)

    api_key = api_key[0]

    fedml.api.login(api_key, compute_node, server, provider)
