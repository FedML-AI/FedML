import click

import fedml.api


@click.command("network", help="Check the Nexus AI backend network connectivity")
@click.help_option("--help", "-h")
@click.option(
    "--open", "-o", default=None, is_flag=True, help="check the connection to open.fedml.ai.",
)
@click.option(
    "--s3", "-s", default=None, is_flag=True, help="check the connection to S3 server.",
)
@click.option(
    "--mqtt", "-m", default=None, is_flag=True, help="check the connection to mqtt.fedml.ai (1883).",
)
@click.option(

    "--mqtt_daemon", "-d", default=None, is_flag=True,
    help="check the connection to mqtt.fedml.ai (1883) with loop mode.",
)
@click.option(
    "--mqtt_s3_backend_server", "-msbs", default=None, is_flag=True,
    help="check the connection to mqtt.fedml.ai (1883) as mqtt+s3 server.",
)
@click.option(
    "--mqtt_s3_backend_client", "-msbc", default=None, is_flag=True,
    help="check the connection to mqtt.fedml.ai (1883) as mqtt+s3 client.",
)
@click.option(
    "--mqtt_s3_backend_run_id", "-rid", type=str, default="fedml_diag_9988", help="mqtt+s3 run id.",
)
def fedml_diagnosis(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
                    mqtt_s3_backend_run_id):
    fedml.api.fedml_diagnosis(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
                              mqtt_s3_backend_run_id)
