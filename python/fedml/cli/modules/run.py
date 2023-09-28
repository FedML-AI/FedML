import click

import fedml.api

@click.group("run")
@click.help_option("--help", "-h")
def fedml_run():
    """
    Run commands into cluster
    """
    pass


def validate_commands(ctx, param, value):
    if not value:
        raise click.BadParameter("At least one command must be provided.")
    return value


@fedml_run.command("run", help="Run command in cluster")
@click.help_option("--help", "-h")
@click.argument("command" "cmd", nargs=-1, callback=validate_commands)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="specify version of MLOps platform. It should be dev, test or release",
)
def run(commands, api_key, version):
    fedml.api.run_command(commands, version=version, api_key=api_key)

