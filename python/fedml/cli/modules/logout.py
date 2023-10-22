import click

import fedml.api


@click.command("logout", help="Logout from the FedML Nexus AI Platform")
@click.help_option("--help", "-h")
@click.option(
    "--computing", "-c", default=None, is_flag=True, help="Logout from the FedML general computing device.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="Logout from the FedML on-premise federated-learning server.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="Logout from which version of FedML® Nexus AI Platform. It should be dev, test or release.",
)
def fedml_logout(computing, server, version):
    fedml.set_env_version(version)

    fedml.api.logout(computing, server)
