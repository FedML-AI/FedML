import logging
import os
import traceback

import fedml

from ..core import mlops


def pre_setup(args):
    mlops.pre_setup(args)


def init(args, should_init_logs=True):
    mlops.init(args, should_init_logs=should_init_logs)


def event(event_name, event_started=True, event_value=None, event_edge_id=None):
    logging.info(f"FedMLDebug edge_id = {event_edge_id}, event_name = {event_name}, START = {event_started}")
    mlops.event(event_name, event_started, event_value, event_edge_id)


def log(metrics: dict, step: int = None, customized_step_key: str = None, commit: bool = True):
    mlops.log(metrics, step=step, customized_step_key=customized_step_key, commit=commit)


def log_llm_record(metrics: dict, version="release") -> None:
    mlops.log_llm_record(metrics, version)


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


def log_training_model_net_info(model_net, dummy_input_tensor):
    return mlops.log_training_model_net_info(model_net, dummy_input_tensor)


def log_training_model_input_info(input_sizes, input_types):
    return mlops.log_training_model_input_info(input_sizes, input_types)


def get_training_model_input_info(training_model_net_url, s3_config):
    return mlops.get_training_model_input_info(training_model_net_url, s3_config)


def log_client_model_info(round_index, total_rounds, model_url):
    mlops.log_client_model_info(round_index, total_rounds, model_url)


def log_sys_perf(sys_args=None):
    try:
        mlops.log_sys_perf(sys_args)
    except Exception as e:
        logging.debug("excpetions when logging sys perf: {}".format(traceback.format_exc()))


def stop_sys_perf():
    try:
        mlops.stop_sys_perf()
    except Exception as e:
        logging.debug("excpetions when stopping sys perf: {}".format(traceback.format_exc()))


def log_server_payload(run_id, edge_id, payload):
    mlops.log_server_payload(run_id, edge_id, payload)


def log_print_init():
    mlops.log_print_start()


def log_print_cleanup():
    mlops.log_print_end()


ARTIFACT_TYPE_GENERAL = 1  # general file
ARTIFACT_TYPE_MODEL = 2  # model file
ARTIFACT_TYPE_DATASET = 3  # dataset file
ARTIFACT_TYPE_SOURCE = 4  # source code
ARTIFACT_TYPE_LOG = 5  # log file

ARTIFACT_TYPE_NAME_GENERAL = "general"  # general file
ARTIFACT_TYPE_NAME_MODEL = "model"  # model file
ARTIFACT_TYPE_NAME_DATASET = "dataset"  # dataset file
ARTIFACT_TYPE_NAME_SOURCE = "source code"  # source code
ARTIFACT_TYPE_NAME_LOG = "log"  # log file

artifact_type_map = {ARTIFACT_TYPE_NAME_GENERAL: ARTIFACT_TYPE_GENERAL,
                     ARTIFACT_TYPE_NAME_MODEL: ARTIFACT_TYPE_MODEL,
                     ARTIFACT_TYPE_NAME_DATASET: ARTIFACT_TYPE_DATASET,
                     ARTIFACT_TYPE_NAME_SOURCE: ARTIFACT_TYPE_SOURCE,
                     ARTIFACT_TYPE_NAME_LOG: ARTIFACT_TYPE_LOG}


class Artifact:
    def __init__(self, name="", type=ARTIFACT_TYPE_NAME_GENERAL):
        self.artifact_name = name
        self.artifact_type_name = type
        self.artifact_type = artifact_type_map[type]
        self.artifact_desc = ""
        self.artifact_files = list()
        self.artifact_dirs = list()
        self.ext_info = dict()

    def add_file(self, file_path):
        if os.path.exists(file_path):
            self.artifact_files.append(file_path)

    def add_dir(self, dir_path):
        if os.path.exists(dir_path):
            self.artifact_dirs.append(dir_path)

    def set_ext_info(self, ext_info_dict):
        self.ext_info = ext_info_dict


def log_artifact(artifact: Artifact, version=None, run_id=None, edge_id=None, async_upload=True):
    mlops.log_artifact(artifact, version=version, run_id=run_id, edge_id=edge_id, async_upload=async_upload)


def log_model(model_name, model_file_path, version=None):
    mlops.log_model(model_name, model_file_path, version=version)


def log_metric(
        metrics: dict, step: int = None, customized_step_key: str = None,
        commit: bool = True, run_id=None, edge_id=None
):
    mlops.log_metric(
        metrics, step=step, customized_step_key=customized_step_key, commit=commit,
        run_id=run_id, edge_id=edge_id
    )


from ..computing.scheduler.slave.client_constants import ClientConstants
from ..computing.scheduler.master.server_constants import ServerConstants

__all__ = [
    "ClientConstants",
    "ServerConstants",
]
