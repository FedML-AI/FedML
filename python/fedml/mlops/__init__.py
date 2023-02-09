from ..core import mlops


def pre_setup(args):
    mlops.pre_setup(args)


def init(args):
    mlops.init(args)


def event(event_name, event_started=True, event_value=None, event_edge_id=None):
    mlops.event(event_name, event_started, event_value, event_edge_id)


def log(metrics):
    mlops.log(metrics)


# status list : ClientStatus
def log_training_status(status, run_id=None):
    mlops.log_training_status(status, run_id)


def log_training_finished_status(run_id=None):
    mlops.log_training_finished_status(run_id)


def log_training_failed_status(run_id=None):
    mlops.log_training_failed_status(run_id)


# status list : ServerStatus
def log_aggregation_status(status, run_id=None):
    mlops.log_aggregation_status(status, run_id)


def log_aggregation_finished_status(run_id=None):
    mlops.log_aggregation_finished_status(run_id)


def send_exit_train_msg(run_id=None):
    mlops.send_exit_train_msg(run_id)


def log_aggregation_failed_status(run_id=None):
    mlops.log_aggregation_failed_status(run_id)


def log_round_info(total_rounds, round_index):
    mlops.log_round_info(total_rounds, round_index)


def log_aggregated_model_info(round_index, model_url):
    mlops.log_aggregated_model_info(round_index, model_url)


def log_training_model_net_info(model_net):
    mlops.log_training_model_net_info(model_net)


def log_client_model_info(round_index, total_rounds, model_url):
    mlops.log_client_model_info(round_index, total_rounds, model_url)


def log_sys_perf(sys_args=None):
    mlops.log_sys_perf(sys_args)


from ..cli.edge_deployment.client_constants import ClientConstants
from ..cli.server_deployment.server_constants import ServerConstants

__all__ = [
    "ClientConstants",
    "ServerConstants",
]
