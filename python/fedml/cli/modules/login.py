
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
    help="Login which version of FedML® Nexus AI Platform. It should be dev, test or release.",
)
@click.option(
    "--computing", "-c", default=None, is_flag=True,
    help="Login as the FedML general computing device, which is the default login option."
         "You can not specify the option -c and -s simultaneously.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True,
    help="Login as the FedML on-premise federated-learning server."
         "You can not specify the option -c and -s simultaneously.",
)
@click.option(
    "--supplier", "-p", default=None, is_flag=True,
    help="Login as the FedML supplier computing device which will connect to a pay-in account."
         "You can specify the option -p and -c simultaneously, but you can not specify -p and -s simultaneously.",
)
def fedml_login(api_key, version, computing, server, supplier):
    fedml.set_env_version(version)

    api_key = api_key[0]
    
    fedml.api.login(api_key, computing, server, supplier)
