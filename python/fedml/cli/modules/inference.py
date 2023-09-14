import click

from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards


@click.group("inference")
def inference():
    """
    Inference models.
    """
    pass


@inference.command("query", help="Query inference parameters for specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def query_model_infer(name):
    inference_output_url, model_metadata, model_config = FedMLModelCards.get_instance().query_model(name)
    if inference_output_url != "":
        click.echo("Query model {} successfully.".format(name))
        click.echo("infer url: {}.".format(inference_output_url))
        click.echo("model metadata: {}.".format(model_metadata))
        click.echo("model config: {}.".format(model_config))
    else:
        click.echo("Failed to query model {}.".format(name))


@inference.command("run", help="Run inference action for specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--data", "-d", type=str, help="input data for model inference.",
)
def run_model_infer(name, data):
    infer_out_json = FedMLModelCards.get_instance().inference_model(name, data)
    if infer_out_json != "":
        click.echo("Inference model {} successfully.".format(name))
        click.echo("Result: {}.".format(infer_out_json))
    else:
        click.echo("Failed to inference model {}.".format(name))
