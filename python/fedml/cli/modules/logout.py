import click

import fedml.api


@click.command("logout", help="Logout from the FedML® Nexus AI Platform")
@click.help_option("--help", "-h")
@click.option(
    "--computing", "-c", default=None, is_flag=True, help="Logout from the FedML general compute node.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="Logout from the the FedML on-premise parameter server (PS).",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="Logout which backend environment version of FedML® Nexus AI Platform. It should be dev, test, or release.",
)
def fedml_logout(computing, server, version):
    fedml.set_env_version(version)

    fedml.api.logout(computing, server)
