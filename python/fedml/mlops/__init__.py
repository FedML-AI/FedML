from ..core import mlops
from ..cli.edge_deployment.client_constants import ClientConstants as ClientStatus
from ..cli.server_deployment.server_constants import ServerConstants as ServerStatus


def init(args):
    mlops.init(args)


def event(event_name, event_started=True, event_value=None, event_edge_id=None):
    mlops.event(event_name, event_started, event_value, event_edge_id)


def log(metrics):
    mlops.log(metrics)


# status list : ClientStatus
def log_training_status(status):
    mlops.log_training_status(status)


# status list : ServerStatus
def log_aggregation_status(status):
    mlops.log_aggregation_status(status)


def log_round_info(total_rounds, round_index):
    mlops.log_round_info(total_rounds, round_index)

