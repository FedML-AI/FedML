import os
from enum import Enum

import click

import fedml.api
from fedml.api.modules.utils import authenticate
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.computing.scheduler.scheduler_core.general_constants import MarketplaceType


@click.command("login", help="Login the TensorOpera® AI Platform")
@click.help_option("--help", "-h")
@click.argument("api_key", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="Login which backend environment version of TensorOpera® AI Platform. It should be dev, test, or release.",
)
@click.option(
    "--compute_node", "-c", default=None, is_flag=True,
    help="Login as the general compute node in FEDML Nexus AI compute network. This is enabled by default. "
         "After login, you can view and manage the device in the TensorOpera® AI Platform: https://tensoropera.ai/gpu/local?label=Private. "
         "It can be grouped as a cluster and then you can use TensorOpera®Launch to schedule any job (training, deployment, federated learning) to it. "
         "You can not specify the option -c and -s simultaneously.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True,
    help="Login as the FedML on-premise parameter server (PS). It can be used for PS-based training paradigms, such as distributed training, cross-cloud training, and federated-learning. "
         "You can not specify the option -c and -s simultaneously for a single environment.",
)
@click.option(
    "--provider", "-p", default=None, is_flag=True,
    help="Login as the FedML compute node (GPU) provider (supplier). This is used by Nexus AI Platform - Share and Earn: https://tensoropera.ai/share-and-earn. You can share your GPUs in this way and earn money. "
         "You can specify the option -p and -c simultaneously (can be used as provider for others as well compute node for your own jobs), but you can not specify -p and -s simultaneously.",
)
@click.option(
    "--deploy_worker_num", "-dpn", default=1, type=int,
    help="Deploy worker number will be started when logged in successfully.",
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
    "--master_inference_gateway_port",
    "-mgp",
    type=int,
    default=ServerConstants.MODEL_INFERENCE_DEFAULT_PORT,
    help="The port for master inference gateway.",
)
@click.option(
    "--worker_inference_proxy_port",
    "-wpp",
    type=int,
    default=ClientConstants.LOCAL_CLIENT_API_PORT,
    help="The port for worker inference proxy.",
)
@click.option(
    "--worker_connection_type",
    "-wct",
    type=str,
    default=ClientConstants.WORKER_CONNECTIVITY_TYPE_DEFAULT,
    help="The connection type for worker inference proxy.",
)
@click.option(
    "--marketplace_type",
    "-mpt",
    type=click.Choice([marketplace_type for marketplace_type in MarketplaceType.__members__]),
    default=MarketplaceType.SECURE.name,
    help="Specify the marketplace type: 'SECURE' for Secure Cloud or 'COMMUNITY' for Community Cloud. "
         "Defaults to Secure Cloud.",
)
@click.option(
    "--price_per_hour",
    "-pph",
    type=click.FLOAT,
    default=0.0,
    help="Enter the price per GPU per hour as a non-negative floating-point number between 0.0 and 1000.0. For "
         "example, if the cost of using an H100 node"
         "for one hour is $1.5 per GPU, then you would input 1.5. Do not multiply this number by the total number of "
         "GPUs in the node, as the system will automatically detect the number of GPUs and include it in the cost "
         "calculation. Default is 0.0."
         "Optionally, you can also set this price later through supplier page on the TensorOpera® AI Platform."
)
@click.option(
    "--name",
    "-n",
    type=str,
    default="",
    help="Name of the node.",
)
def fedml_login(
        api_key, version, compute_node, server, provider, deploy_worker_num,
        local_on_premise_platform, local_on_premise_platform_port,
        master_inference_gateway_port, worker_inference_proxy_port, worker_connection_type, marketplace_type,
        price_per_hour, name
):
    fedml.set_env_version(version)
    fedml.set_local_on_premise_platform_host(local_on_premise_platform)
    fedml.set_local_on_premise_platform_port(local_on_premise_platform_port)

    try:
        price_per_hour = float(price_per_hour)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="price_per_hour")

    __validate_mpt_pph(marketplace_type, price_per_hour)

    api_key = api_key[0] if len(api_key) > 0 else None
    try:
        authenticate(api_key)
    except SystemExit as e:
        print(f"{str(e)}\n")
        print(f"Maybe you are using account id to login, we will try to login with account {api_key}.")
        pass
    os.environ["FEDML_MODEL_WORKER_NUM"] = str(deploy_worker_num)
    fedml.api.login(api_key, compute_node, server, provider, master_inference_gateway_port,
                    worker_inference_proxy_port, worker_connection_type, marketplace_type, price_per_hour, name)


def __validate_mpt_pph(marketplace_type, price_per_hour):
    try:
        MarketplaceType.from_str(marketplace_type)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="marketplace_type")

    if price_per_hour < 0 or price_per_hour > 1000:
        raise click.BadParameter(f"Price per hour should be a non-negative float ranging between 0 and 1000. Current "
                                 f"input value {price_per_hour} is not valid", param_hint="price_per_hour")
