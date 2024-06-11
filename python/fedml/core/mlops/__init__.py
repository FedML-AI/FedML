import json
import logging
import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
from multiprocessing import Process

import requests

import fedml
from fedml import constants
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.core.mlops.mlops_configs import MLOpsConfigs
from .mlops_constants import MLOpsConstants

from ...constants import FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_TRAINING_PLATFORM_SIMULATION_TYPE

from .mlops_metrics import MLOpsMetrics
from .mlops_profiler_event import MLOpsProfilerEvent
from .mlops_runtime_log import MLOpsRuntimeLog
from .mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from .mlops_runtime_log_daemon import MLOpsRuntimeLogProcessor
from .mlops_status import MLOpsStatus
from .mlops_utils import MLOpsUtils, MLOpsLoggingUtils, LogFile
from .system_stats import SysStats
from ..distributed.communication.mqtt.mqtt_manager import MqttManager
from ..distributed.communication.s3.remote_storage import S3Storage
from ...computing.scheduler.master.server_constants import ServerConstants
from ...computing.scheduler.slave.client_constants import ClientConstants
from ...computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
from .mlops_utils import MLOpsUtils
from .mlops_constants import MLOpsConstants
from ...computing.scheduler.master.master_protocol_manager import FedMLLaunchMasterProtocolManager
from ...computing.scheduler.scheduler_core.account_manager import FedMLAccountManager


FEDML_MLOPS_API_RESPONSE_SUCCESS_CODE = "SUCCESS"

__all__ = [
    "MLOpsMetrics",
    "MLOpsProfilerEvent",
    "SysStats",
    "MLOpsStatus",
    "MLOpsRuntimeLog",
    "MLOpsRuntimeLogProcessor",
    "MLOpsRuntimeLogDaemon",
    "log_aggregation_failed_status",
    "log_training_failed_status",
    "log_endpoint_status",
    "MLOpsConfigs",
    "sync_deploy_id"
]


class MLOpsStore:
    mlops_args = None
    mlops_project_id: int = None
    mlops_run_id = None
    mlops_edge_id = None
    mlops_log_metrics_steps = 0
    mlops_log_metrics = dict()
    mlops_log_records = dict()
    mlops_log_round_info = dict()
    mlops_log_client_training_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING
    mlops_log_server_training_status = ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING
    mlops_log_round_start_time = 0.0
    mlops_log_metrics_lock = None
    mlops_log_records_lock = None
    mlops_log_mqtt_mgr = None
    mlops_log_mqtt_lock = None
    mlops_log_mqtt_is_connected = False
    mlops_log_agent_config = dict()
    mlops_metrics = None
    mlops_event = None
    mlops_bind_result = False
    server_agent_id = None
    current_parrot_process = None
    mlops_run_status_callback = None

    METRIC_NAME_X_AXIS = "x_axis_keys"
    METRIC_NAME_Y_AXIS = "y_axis_keys"
    METRICS_X_AXIS_TAG_DEFAULT = "step"
    METRICS_X_AXIS_TAG_TIMESTAMP = "timestamp"
    METRICS_X_AXIS_TAG_LIST = ["index", "idx", "iteration", "iter", "round", "round_index", "round_idx",
                               "round-index", "round-idx", "epoch", "round", "step", "timestamp"]

    def __init__(self):
        pass


def pre_setup(args):
    MLOpsStore.mlops_args = args


def init(args, should_init_logs=True):
    MLOpsStore.mlops_args = args
    if not mlops_parrot_enabled(args):
        if not hasattr(args, "config_version"):
            args.config_version = "release"
        if should_init_logs:
            MLOpsRuntimeLog.get_instance(args).init_logs()
        fetch_config(args, args.config_version)
        return
    else:
        if hasattr(args, "simulator_daemon"):
            # Bind local device as simulation device on FedML® Nexus AI Platform
            setattr(args, "using_mlops", True)
            setattr(args, "rank", 1)
            MLOpsStore.mlops_bind_result = bind_simulation_device(args, args.user)
            return

    project_name = None
    api_key = None
    run_name = None
    if hasattr(args, "mlops_project_name"):
        project_name = args.mlops_project_name
    if hasattr(args, "mlops_api_key"):
        api_key = args.mlops_api_key
    if hasattr(args, "mlops_run_name"):
        run_name = args.mlops_run_name
    if project_name is None or api_key is None:
        raise Exception("Please check mlops_project_name and mlops_api_key params.")

    # Bind local device as simulation device on FedML® Nexus AI Platform
    setattr(args, "using_mlops", True)
    setattr(args, "rank", 1)
    MLOpsStore.mlops_bind_result = bind_simulation_device(args, api_key, args.config_version)
    if not MLOpsStore.mlops_bind_result:
        setattr(args, "using_mlops", False)
        if should_init_logs:
            MLOpsRuntimeLog.get_instance(args).init_logs()
        return

    # Init project and run
    result_project, project_id = create_project(project_name, api_key)
    if result_project:
        result_run, run_id = create_run(project_id, api_key, run_name)
        if result_run:
            MLOpsStore.mlops_project_id = project_id
            MLOpsStore.mlops_run_id = run_id
    if result_project is False or result_run is False:
        print("Failed to init project and run.")
        return

    # Init runtime logs
    if should_init_logs:
        init_logs(MLOpsStore.mlops_args, MLOpsStore.mlops_edge_id)
    logging.info("mlops.init args {}".format(MLOpsStore.mlops_args))

    # Save current process id
    MLOpsStore.current_parrot_process = os.getpid()

    # Start simulator login process as daemon
    mlops_simulator_login(api_key, MLOpsStore.mlops_run_id)


def event(event_name, event_started=True, event_value=None, event_edge_id=None):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()

    if event_started:
        MLOpsStore.mlops_event.log_event_started(event_name, event_value, event_edge_id)
    else:
        MLOpsStore.mlops_event.log_event_ended(event_name, event_value, event_edge_id)


def log(metrics: dict, step: int = None, customized_step_key: str = None, commit: bool = True, is_endpoint_metric=False):
    if MLOpsStore.mlops_args is None or fedml._global_training_type == constants.FEDML_TRAINING_PLATFORM_CROSS_CLOUD:
        log_metric(metrics, step=step, customized_step_key=customized_step_key, commit=commit,
                   is_endpoint_metric=is_endpoint_metric)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    log_metric(metrics, step=step, customized_step_key=customized_step_key, commit=commit,
               run_id=MLOpsStore.mlops_run_id, edge_id=MLOpsStore.mlops_edge_id,
               is_endpoint_metric=is_endpoint_metric)


def log_endpoint(metrics: dict, step: int = None, customized_step_key: str = None, commit: bool = True):
    if MLOpsStore.mlops_args is None or fedml._global_training_type == constants.FEDML_TRAINING_PLATFORM_CROSS_CLOUD:
        log_metric(metrics, step=step, customized_step_key=customized_step_key, commit=commit, is_endpoint_metric=True)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    log_metric(metrics, step=step, customized_step_key=customized_step_key, commit=commit,
               run_id=MLOpsStore.mlops_run_id, edge_id=MLOpsStore.mlops_edge_id, is_endpoint_metric=True)


def log_llm_record(metrics: dict, version="release", commit: bool = True) -> None:
    if MLOpsStore.mlops_log_records_lock is None:
        MLOpsStore.mlops_log_records_lock = threading.Lock()

    MLOpsStore.mlops_log_records_lock.acquire()
    for k, v in metrics.items():
        k = str(k).replace("/", "_")
        if k.startswith("round"):
            k = "round_idx"

        MLOpsStore.mlops_log_records[k] = v
    MLOpsStore.mlops_log_records["run_id"] = str(MLOpsStore.mlops_run_id)
    MLOpsStore.mlops_log_records["timestamp"] = float(time.time_ns() / 1000 / 1000 * 1.0)
    MLOpsStore.mlops_log_records_lock.release()

    logging.info("log records {}".format(json.dumps(MLOpsStore.mlops_log_records)))

    if len(MLOpsStore.mlops_log_agent_config) == 0:
        mqtt_config, s3_config, mlops_config, docker_config = MLOpsConfigs.fetch_all_configs()
        service_config = dict()
        service_config["mqtt_config"] = mqtt_config
        service_config["s3_config"] = s3_config
        service_config["ml_ops_config"] = mlops_config
        service_config["docker_config"] = docker_config
        MLOpsStore.mlops_log_agent_config = service_config

    if commit:
        setup_log_mqtt_mgr()
        MLOpsStore.mlops_log_records_lock.acquire()
        MLOpsStore.mlops_metrics.report_llm_record(MLOpsStore.mlops_log_records)
        MLOpsStore.mlops_log_records.clear()
        MLOpsStore.mlops_log_records_lock.release()


def log_training_status(status, run_id=None, edge_id=None, is_from_model=False, enable_broadcast=False):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    else:
        run_id = MLOpsStore.mlops_run_id
    if edge_id is not None:
        MLOpsStore.mlops_edge_id = edge_id
    else:
        edge_id = MLOpsStore.mlops_edge_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()
    if MLOpsStore.mlops_metrics is None:
        return
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        MLOpsStore.mlops_metrics.report_client_training_status(
            edge_id, status, is_from_model=is_from_model, run_id=run_id)
    else:
        MLOpsStore.mlops_metrics.report_client_id_status(
            edge_id, status, is_from_model=is_from_model, run_id=run_id)

        if enable_broadcast:
            MLOpsStore.mlops_metrics.report_client_training_status(
                edge_id, status, is_from_model=is_from_model, run_id=run_id)


def log_aggregation_status(status, run_id=None, edge_id=None):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    else:
        run_id = MLOpsStore.mlops_run_id
    if edge_id is not None:
        MLOpsStore.mlops_edge_id = edge_id
    else:
        edge_id = MLOpsStore.mlops_edge_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        device_role = "simulator"
    else:
        device_role = "server"
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        MLOpsStore.mlops_metrics.report_server_training_status(
            run_id, status, role=device_role, edge_id=edge_id)
        sys_utils.save_simulator_process(ClientConstants.get_data_dir(),
                                         ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME, os.getpid(),
                                         str(run_id),
                                         run_status=status)

        # Start log processor for current run
        if status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
            MLOpsRuntimeLogDaemon.get_instance(MLOpsStore.mlops_args).stop_log_processor(
                run_id, edge_id)
    else:
        MLOpsStore.mlops_metrics.report_server_id_status(
            run_id, status, edge_id=edge_id,
            server_id=edge_id, server_agent_id=edge_id
        )


def log_training_finished_status(run_id=None, is_from_model=False, edge_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_training_status(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED, run_id=run_id, edge_id=edge_id)
        time.sleep(2)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    else:
        run_id = MLOpsStore.mlops_run_id
    if edge_id is not None:
        MLOpsStore.mlops_edge_id = edge_id
    else:
        edge_id = MLOpsStore.mlops_edge_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()
    if MLOpsStore.mlops_metrics is not None:
        MLOpsStore.mlops_metrics.report_client_id_status(
            edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
            is_from_model=is_from_model, run_id=run_id)


def send_exit_train_msg(run_id=None):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()
    if MLOpsStore.mlops_metrics is not None:
        MLOpsStore.mlops_metrics.client_send_exit_train_msg(
            MLOpsStore.mlops_run_id if run_id is None else run_id,
            MLOpsStore.mlops_edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)


def log_training_failed_status(run_id=None, edge_id=None, is_from_model=False, enable_broadcast=False):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_training_status(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED, run_id=run_id, edge_id=edge_id)
        time.sleep(2)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    else:
        run_id = MLOpsStore.mlops_run_id
    if edge_id is not None:
        MLOpsStore.mlops_edge_id = edge_id
    else:
        edge_id = MLOpsStore.mlops_edge_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()

    if MLOpsStore.mlops_metrics is None:
        return

    MLOpsStore.mlops_metrics.report_client_id_status(
        edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED, is_from_model=is_from_model, run_id=run_id)
    if enable_broadcast:
        MLOpsStore.mlops_metrics.report_client_training_status(
            edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED, is_from_model=is_from_model, run_id=run_id)


def log_aggregation_finished_status(run_id=None, edge_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_aggregation_status(ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED, run_id=run_id, edge_id=edge_id)
        time.sleep(15)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    else:
        run_id = MLOpsStore.mlops_run_id
    if edge_id is not None:
        MLOpsStore.mlops_edge_id = edge_id
    else:
        edge_id = MLOpsStore.mlops_edge_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()

    if MLOpsStore.mlops_metrics is None:
        return

    MLOpsStore.mlops_metrics.report_server_id_status(
        run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
        edge_id=edge_id, server_id=edge_id, server_agent_id=edge_id
    )


def log_aggregation_failed_status(run_id=None, edge_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_aggregation_status(ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, run_id=run_id, edge_id=edge_id)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    else:
        run_id = MLOpsStore.mlops_run_id
    if edge_id is not None:
        MLOpsStore.mlops_edge_id = edge_id
    else:
        edge_id = MLOpsStore.mlops_edge_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()

    if MLOpsStore.mlops_metrics is None:
        return

    MLOpsStore.mlops_metrics.report_server_id_status(
        run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=edge_id,
        server_id=edge_id, server_agent_id=edge_id
    )


def log_aggregation_exception_status(run_id=None, edge_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_aggregation_status(ServerConstants.MSG_MLOPS_SERVER_STATUS_EXCEPTION, run_id=run_id, edge_id=edge_id)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    else:
        run_id = MLOpsStore.mlops_run_id
    if edge_id is not None:
        MLOpsStore.mlops_edge_id = edge_id
    else:
        edge_id = MLOpsStore.mlops_edge_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()

    if MLOpsStore.mlops_metrics is None:
        return

    MLOpsStore.mlops_metrics.report_server_id_status(
        run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_EXCEPTION, edge_id=edge_id,
        server_id=edge_id, server_agent_id=edge_id
    )


def callback_run_status_changed(topic, payload):
    payload_obj = json.loads(payload)
    run_id = payload_obj.get("run_id", 0)
    run_status = payload_obj.get("status")
    if MLOpsStore.mlops_run_status_callback is not None:
        MLOpsStore.mlops_run_status_callback(run_id, run_status)


# run_status_callback: def run_status_callback(run_id, run_status)
# run_status: FINISHED, FAILED, KILLED, etc.
def register_run_status_callback(run_status_callback):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    # logging.info("log aggregation inner status {}".format(ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED))

    setup_log_mqtt_mgr()

    MLOpsStore.mlops_run_status_callback = run_status_callback

    topic_client_status = "fl_client/flclient_agent_" + str(MLOpsStore.mlops_edge_id) + "/status"
    topic_server_status = "fl_server/flserver_agent_" + str(MLOpsStore.mlops_edge_id) + "/status"
    MLOpsStore.mlops_log_mqtt_mgr.add_message_listener(topic_client_status, callback_run_status_changed)
    MLOpsStore.mlops_log_mqtt_mgr.add_message_listener(topic_server_status, callback_run_status_changed)
    MLOpsStore.mlops_log_mqtt_mgr.subscribe_msg(topic_client_status)
    MLOpsStore.mlops_log_mqtt_mgr.subscribe_msg(topic_server_status)


def log_aggregated_model_info(round_index, model_url):
    if model_url is None:
        return
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    logging.info("log aggregated model info {}".format(model_url))

    setup_log_mqtt_mgr()
    model_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "round_idx": round_index,
        "global_aggregated_model_s3_address": model_url,
    }

    if MLOpsStore.mlops_metrics is None:
        return

    MLOpsStore.mlops_metrics.report_aggregated_model_info(model_info)


def log_training_model_net_info(model_net, dummy_input_tensor):
    if model_net is None:
        return None
    if not mlops_enabled(MLOpsStore.mlops_args):
        return None

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return None

    s3_config = MLOpsStore.mlops_log_agent_config.get("s3_config", None)
    if s3_config is None:
        return None
    s3_client = S3Storage(s3_config)
    model_key = "fedml-model-net-run-{}-{}".format(str(MLOpsStore.mlops_run_id), str(uuid.uuid4()))
    model_url = s3_client.write_model_net(model_key, model_net,
                                          dummy_input_tensor, ClientConstants.get_model_cache_dir())

    logging.info("log training model net info {}".format(model_url))

    setup_log_mqtt_mgr()
    model_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "training_model_net_s3_address": model_url,
    }
    if MLOpsStore.mlops_metrics is not None:
        MLOpsStore.mlops_metrics.report_training_model_net_info(model_info)
    return model_url


def log_training_model_input_info(input_sizes, input_types):
    if input_sizes is None or input_types is None:
        return None
    if not mlops_enabled(MLOpsStore.mlops_args):
        return None

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return None

    s3_config = MLOpsStore.mlops_log_agent_config.get("s3_config", None)
    if s3_config is None:
        return None
    s3_client = S3Storage(s3_config)
    model_key = "fedml-model-input-run-{}".format(str(MLOpsStore.mlops_run_id))
    model_input_url = s3_client.write_model_input(model_key, input_sizes, input_types,
                                                  ClientConstants.get_model_cache_dir())
    logging.info(f"training model input: {model_input_url}")

    return model_input_url


def get_training_model_input_info(training_model_net_url, s3_config):
    if s3_config is None:
        return None, None

    run_id = str(training_model_net_url).split("fedml-model-net-run-")[1].split("-")[0]
    model_key = f"fedml-model-input-run-{run_id}"
    s3_client = S3Storage(s3_config)
    input_size, input_type = s3_client.read_model_input(model_key, ClientConstants.get_model_cache_dir())
    logging.info(f"training model input size: {input_size}, input type: {input_type}")
    return input_size, input_type


def log_client_model_info(round_index, total_rounds, model_url):
    if model_url is None:
        return
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    logging.info("log client model info {}".format(model_url))

    setup_log_mqtt_mgr()
    model_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "edge_id": MLOpsStore.mlops_edge_id,
        "round_idx": round_index,
        "client_model_s3_address": model_url,
    }
    if MLOpsStore.mlops_metrics is not None:
        MLOpsStore.mlops_metrics.report_client_model_info(model_info)

    FedMLClientDataInterface.get_instance().save_running_job(MLOpsStore.mlops_run_id, MLOpsStore.mlops_edge_id,
                                                             round_index,
                                                             total_rounds,
                                                             "Running")


def log_sys_perf(sys_args=None):
    if not mlops_enabled(sys_args):
        return

    if sys_args is not None:
        MLOpsStore.mlops_args = sys_args

    sys_metrics = MLOpsMetrics()
    sys_metrics.report_sys_perf(MLOpsStore.mlops_args,
                                MLOpsStore.mlops_log_agent_config["mqtt_config"])


def stop_sys_perf():
    metrics = MLOpsMetrics()
    metrics.stop_sys_perf()


def log_server_payload(run_id, edge_id, payload):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not mlops_enabled(MLOpsStore.mlops_args):
        return False

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()
    topic = "fedml_{}_{}".format(run_id, edge_id)
    logging.info("log json message, topic {}, payload {}.".format(topic, payload))
    if MLOpsStore.mlops_metrics is None:
        return
    MLOpsStore.mlops_metrics.report_json_message(topic, payload)


def log_print_start():
    fedml_args = get_fedml_args()

    setattr(fedml_args, "using_mlops", True)
    MLOpsRuntimeLogDaemon.get_instance(fedml_args).start_log_processor(fedml_args.run_id, fedml_args.run_device_id)


def log_print_end():
    fedml_args = get_fedml_args()

    setattr(fedml_args, "using_mlops", True)
    MLOpsRuntimeLogDaemon.get_instance(fedml_args).stop_log_processor(fedml_args.run_id, fedml_args.run_device_id)


def get_fedml_args():
    # init FedML framework
    fedml._global_training_type = constants.FEDML_TRAINING_PLATFORM_CROSS_CLOUD
    fedml._global_comm_backend = ""
    fedml_args = fedml.init(check_env=False, should_init_logs=False)
    fedml_args.version = fedml.get_env_version()
    fedml_args.config_version = fedml.get_env_version()
    fedml_args.using_mlops = True
    return fedml_args


def push_artifact_to_s3(artifact: fedml.mlops.Artifact, version="release", show_progress=True):
    args = {"config_version": version}
    _, s3_config, _, _ = MLOpsConfigs.fetch_all_configs()
    s3_storage = S3Storage(s3_config)
    artifact_dst_key = f"{artifact.artifact_name}_{artifact.artifact_type_name}"
    artifact_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "artifacts")
    artifact_archive_name = os.path.join(artifact_dir, artifact_dst_key)
    os.makedirs(artifact_archive_name, exist_ok=True)

    for artifact_item in artifact.artifact_files:
        artifact_base_name = os.path.basename(artifact_item)
        shutil.copyfile(artifact_item, os.path.join(artifact_archive_name, artifact_base_name))

    for artifact_item in artifact.artifact_dirs:
        artifact_base_name = os.path.basename(artifact_item)
        dst_dir = os.path.join(artifact_archive_name, artifact_base_name)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir, ignore_errors=True)
        shutil.copytree(artifact_item, os.path.join(artifact_archive_name, artifact_base_name),
                        ignore_dangling_symlinks=True)

    shutil.make_archive(
        artifact_archive_name,
        "zip",
        root_dir=artifact_dir,
        base_dir=artifact_dst_key,
    )
    artifact_archive_zip_file = artifact_archive_name + ".zip"
    artifact_storage_url = ""
    try:
        artifact_dst_key = f"{artifact_dst_key}.zip"
        artifact_storage_url = s3_storage.upload_file_with_progress(artifact_archive_zip_file, artifact_dst_key,
                                                                    show_progress=show_progress,
                                                                    out_progress_to_err=True,
                                                                    progress_desc="Submitting your artifact to "
                                                                                  "FedML® Nexus AI Platform")
        artifact_storage_url = str(artifact_storage_url).split("?")[0]
    except Exception as e:
        pass
    return artifact_archive_zip_file, artifact_storage_url


def log_artifact(artifact: fedml.mlops.Artifact, version=None, run_id=None, edge_id=None, async_upload=True):
    if async_upload:
        Process(target=_log_artifact_async, args=(
            artifact, version, run_id, edge_id
        )).start()
        return
    else:
        _log_artifact_sync(artifact, version=version, run_id=run_id, edge_id=edge_id)


def _log_artifact_sync(
        artifact: fedml.mlops.Artifact, version=None, run_id=None, edge_id=None
):
    fedml_args = get_fedml_args()

    artifact_archive_zip_file, artifact_storage_url = push_artifact_to_s3(
        artifact, version=version if version is not None else fedml_args.config_version)

    setup_log_mqtt_mgr()
    if run_id is None:
        run_id = os.getenv('FEDML_CURRENT_RUN_ID', 0)
    if edge_id is None:
        edge_id = os.getenv('FEDML_CURRENT_EDGE_ID', 0)
    timestamp = MLOpsUtils.get_ntp_time()
    if MLOpsStore.mlops_metrics is None:
        return
    MLOpsStore.mlops_metrics.report_artifact_info(run_id, edge_id, artifact.artifact_name, artifact.artifact_type,
                                                  artifact_archive_zip_file, artifact_storage_url,
                                                  artifact.ext_info, artifact.artifact_desc,
                                                  timestamp)


def _log_artifact_async(
        artifact: fedml.mlops.Artifact, version=None, run_id=None, edge_id=None
):
    fedml_args = get_fedml_args()
    fetch_config(fedml_args, version=fedml.get_env_version())
    agent_config = MLOpsStore.mlops_log_agent_config

    artifact_archive_zip_file, artifact_storage_url = push_artifact_to_s3(
        artifact, version=version if version is not None else fedml_args.config_version)

    device_id = str(uuid.uuid4())
    log_artifact_mqtt_mgr = MqttManager(
        agent_config["mqtt_config"]["BROKER_HOST"],
        agent_config["mqtt_config"]["BROKER_PORT"],
        agent_config["mqtt_config"]["MQTT_USER"],
        agent_config["mqtt_config"]["MQTT_PWD"],
        agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
        "FedML_MLOps_Metrics_{}_{}_{}".format(
            device_id, str(edge_id), str(uuid.uuid4()))
    )
    log_artifact_mqtt_mgr.connect()
    log_artifact_mqtt_mgr.loop_start()

    if run_id is None:
        run_id = os.getenv('FEDML_CURRENT_RUN_ID', 0)
    if edge_id is None:
        edge_id = os.getenv('FEDML_CURRENT_EDGE_ID', 0)
    timestamp = MLOpsUtils.get_ntp_time()
    log_artifact_metrics = MLOpsMetrics()
    log_artifact_metrics.set_messenger(log_artifact_mqtt_mgr)
    log_artifact_metrics.report_artifact_info(run_id, edge_id, artifact.artifact_name, artifact.artifact_type,
                                              artifact_archive_zip_file, artifact_storage_url,
                                              artifact.ext_info, artifact.artifact_desc,
                                              timestamp)
    log_artifact_mqtt_mgr.disconnect()
    log_artifact_mqtt_mgr.loop_stop()


def log_model(model_name, model_file_path, version=None):
    model_artifact = fedml.mlops.Artifact(name=model_name, type=fedml.mlops.ARTIFACT_TYPE_NAME_MODEL)
    model_artifact.add_file(model_file_path)
    log_artifact(model_artifact, version=version)


def log_metric(metrics: dict, step: int = None, customized_step_key: str = None, commit: bool = True,
               run_id=None, edge_id=None, is_endpoint_metric=False):
    fedml_args = get_fedml_args()

    if MLOpsStore.mlops_log_metrics_lock is None:
        MLOpsStore.mlops_log_metrics_lock = threading.Lock()

    if customized_step_key is not None:
        customized_step_key = customized_step_key.replace('/', '-')

    if run_id is None:
        run_id = os.getenv('FEDML_CURRENT_RUN_ID', None)
    if edge_id is None:
        edge_id = os.getenv('FEDML_CURRENT_EDGE_ID', None)

    if commit:
        MLOpsStore.mlops_log_metrics_lock.acquire()
        if step is None:
            current_step = MLOpsStore.mlops_log_metrics_steps
        else:
            current_step = step
        log_metrics_obj = _generate_log_metrics(
            metrics, step=current_step, customized_step_key=customized_step_key, run_id=run_id, edge_id=edge_id,
            previous_metrics=MLOpsStore.mlops_log_metrics)
        if log_metrics_obj is None:
            MLOpsStore.mlops_log_metrics_lock.release()
            return
        MLOpsStore.mlops_log_metrics = log_metrics_obj.copy()
        setup_log_mqtt_mgr()
        if is_endpoint_metric:
            MLOpsStore.mlops_metrics.report_endpoint_metric(MLOpsStore.mlops_log_metrics)
        else:
            MLOpsStore.mlops_metrics.report_fedml_train_metric(
                MLOpsStore.mlops_log_metrics, run_id=run_id, is_endpoint=is_endpoint_metric)
        MLOpsStore.mlops_log_metrics.clear()
        if step is None:
            MLOpsStore.mlops_log_metrics_steps = current_step + 1
        MLOpsStore.mlops_log_metrics_lock.release()
    else:
        MLOpsStore.mlops_log_metrics_lock.acquire()
        if step is None:
            current_step = MLOpsStore.mlops_log_metrics_steps
        else:
            current_step = step
        log_metrics_obj = _generate_log_metrics(
            metrics, step=current_step, customized_step_key=customized_step_key, run_id=run_id, edge_id=edge_id,
            previous_metrics=MLOpsStore.mlops_log_metrics)
        if log_metrics_obj is None:
            MLOpsStore.mlops_log_metrics_lock.release()
            return
        MLOpsStore.mlops_log_metrics = log_metrics_obj.copy()
        MLOpsStore.mlops_log_metrics_lock.release()


def log_run_logs(logs_json: dict, run_id=0):
    fedml_args = get_fedml_args()

    setup_log_mqtt_mgr()

    if MLOpsStore.mlops_metrics is not None:
        MLOpsStore.mlops_metrics.report_fedml_run_logs(logs_json, run_id=run_id)


def log_run_log_lines(run_id, device_id, log_list, log_source=None, use_mqtt=False):
    fedml_args = get_fedml_args()

    if use_mqtt:
        setup_log_mqtt_mgr()

    if MLOpsStore.mlops_metrics is not None:
        MLOpsStore.mlops_metrics.report_run_log(
            run_id, device_id, log_list, log_source=log_source, use_mqtt=use_mqtt)


def _append_to_list(list_data, list_item):
    try:
        list_data.index(list_item)
    except Exception as e:
        list_data.append(list_item)

    return list_data


def _generate_log_metrics(metrics: dict, step: int = None, customized_step_key: str = None,
                          run_id=None, edge_id=None, previous_metrics=None):
    if run_id is None:
        run_id = os.getenv('FEDML_CURRENT_RUN_ID', None)
    if edge_id is None:
        edge_id = os.getenv('FEDML_CURRENT_EDGE_ID', None)
    if run_id is None or str(run_id).strip() == "":
        return None

    # Generate default x-axis keys
    log_metrics_obj = dict() if previous_metrics is None else previous_metrics.copy()
    if log_metrics_obj.get(MLOpsStore.METRIC_NAME_X_AXIS, None) is None:
        log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS] = list()
    log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS] = _append_to_list(
            log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS], MLOpsStore.METRICS_X_AXIS_TAG_DEFAULT)
    log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS] = _append_to_list(
        log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS], MLOpsStore.METRICS_X_AXIS_TAG_TIMESTAMP)

    # Generate the metrics for y-axis and the keys for x-axis/y-axis
    if log_metrics_obj.get(MLOpsStore.METRIC_NAME_Y_AXIS, None) is None:
        log_metrics_obj[MLOpsStore.METRIC_NAME_Y_AXIS] = list()
    found_customized_step_key = False
    if customized_step_key is not None:
        customized_step_key = str(customized_step_key).lower()
    for k, v in metrics.items():
        k = str(k).lower().replace('/','-')
        log_metrics_obj[k] = v
        found_x_axis = False
        for x_axis in MLOpsStore.METRICS_X_AXIS_TAG_LIST:
            if k == x_axis:
                log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS] = _append_to_list(
                    log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS], x_axis)
                found_x_axis = True
                break
        if not found_x_axis and k != customized_step_key:
            log_metrics_obj[MLOpsStore.METRIC_NAME_Y_AXIS] = _append_to_list(
                log_metrics_obj[MLOpsStore.METRIC_NAME_Y_AXIS], k)

        if k == customized_step_key:
            found_customized_step_key = True

    # Add the key for x-axis with specific step metric key
    if customized_step_key is not None and found_customized_step_key:
        log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS] = _append_to_list(
            log_metrics_obj[MLOpsStore.METRIC_NAME_X_AXIS], customized_step_key)

    # Generate the x-axis metric with the step value
    if step is not None:
        log_metrics_obj[MLOpsStore.METRICS_X_AXIS_TAG_DEFAULT] = step
    else:
        log_metrics_obj[MLOpsStore.METRICS_X_AXIS_TAG_DEFAULT] = 0

    log_metrics_obj["run_id"] = str(run_id)
    log_metrics_obj[MLOpsStore.METRICS_X_AXIS_TAG_TIMESTAMP] = float(time.time_ns() / 1000 / 1000 * 1.0)

    return log_metrics_obj


def log_mlops_running_logs(artifact: fedml.mlops.Artifact, version=None, run_id=None, edge_id=None,
                           only_push_artifact=False):

    artifact_archive_zip_file, artifact_storage_url = push_artifact_to_s3(
        artifact, version=version if version is not None else fedml.get_env_version(), show_progress=False)

    if only_push_artifact:
        return artifact_storage_url

    setup_log_mqtt_mgr()
    if run_id is None:
        run_id = os.getenv('FEDML_CURRENT_RUN_ID', 0)
    if edge_id is None:
        edge_id = os.getenv('FEDML_CURRENT_EDGE_ID', 0)
    timestamp = MLOpsUtils.get_ntp_time()
    if MLOpsStore.mlops_metrics is not None:
        MLOpsStore.mlops_metrics.report_artifact_info(run_id, edge_id, artifact.artifact_name, artifact.artifact_type,
                                                      artifact_archive_zip_file, artifact_storage_url,
                                                      artifact.ext_info, artifact.artifact_desc,
                                                      timestamp)

    return artifact_storage_url


def log_round_info(total_rounds, round_index):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not mlops_enabled(MLOpsStore.mlops_args):
        return False

    if not MLOpsStore.mlops_bind_result:
        return

    if round_index == -1:
        MLOpsStore.mlops_log_round_start_time = time.time()

    setup_log_mqtt_mgr()
    round_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "round_index": round_index,
        "total_rounds": total_rounds,
        "running_time": round(time.time() - MLOpsStore.mlops_log_round_start_time, 4),
    }
    logging.info("log round info {}".format(round_info))
    if MLOpsStore.mlops_metrics is None:
        return
    MLOpsStore.mlops_metrics.report_server_training_round_info(round_info)


def log_endpoint_status(endpoint_id, status):
    fedml_args = get_fedml_args()

    setup_log_mqtt_mgr()
    run_id = os.getenv('FEDML_CURRENT_RUN_ID', 0)
    edge_id = os.getenv('FEDML_CURRENT_EDGE_ID', 0)
    if MLOpsStore.mlops_metrics is None:
        return
    MLOpsStore.mlops_metrics.report_endpoint_status(
        endpoint_id, status, timestamp=MLOpsUtils.get_ntp_time() * 1000.0)


def create_project(project_name, api_key):
    url_prefix, cert_path = get_request_params(MLOpsStore.mlops_args)
    url = "{}/fedmlOpsServer/projects/createSim".format(url_prefix)
    json_params = {"name": project_name,
                   "userids": api_key,
                   "platform_type": str(FEDML_TRAINING_PLATFORM_SIMULATION_TYPE)}
    if cert_path is not None:
        try:
            requests.session().verify = cert_path
            response = requests.post(
                url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
            )
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            response = requests.post(
                url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
            )
    else:
        response = requests.post(
            url, json=json_params, headers={"Connection": "close"}
        )
    status_code = response.json().get("code")
    if status_code == FEDML_MLOPS_API_RESPONSE_SUCCESS_CODE:
        project_id = response.json().get("data")
        return True, project_id
    else:
        return False, 0


def create_run(project_id, api_key, run_name=None):
    url_prefix, cert_path = get_request_params(MLOpsStore.mlops_args)
    url = "{}/fedmlOpsServer/runs/createSim".format(url_prefix)
    edge_ids = list()
    edge_ids.append(MLOpsStore.mlops_edge_id)
    json_params = {"userids": api_key,
                   "projectid": str(project_id),
                   "edgeids": edge_ids}
    if run_name is not None:
        json_params["name"] = run_name
    if cert_path is not None:
        try:
            requests.session().verify = cert_path
            response = requests.post(
                url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
            )
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            response = requests.post(
                url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "close"}
            )
    else:
        response = requests.post(
            url, json=json_params, headers={"Connection": "close"}
        )
    status_code = response.json().get("code")
    if status_code == FEDML_MLOPS_API_RESPONSE_SUCCESS_CODE:
        run_id = response.json().get("data")
        return True, run_id
    else:
        return False, 0


def get_request_params(args):
    url = fedml._get_backend_service()

    cert_path = None
    if str(url).startswith("https://"):
        cur_source_dir = os.path.dirname(__file__)
        cert_path = os.path.join(
            cur_source_dir, "ssl", "open-" + fedml.get_env_version() + ".fedml.ai_bundle.crt"
        )

    return url, cert_path


def on_log_mqtt_disconnected(mqtt_client_object):
    if MLOpsStore.mlops_log_mqtt_lock is None:
        MLOpsStore.mlops_log_mqtt_lock = threading.Lock()

    MLOpsStore.mlops_log_mqtt_lock.acquire()
    MLOpsStore.mlops_log_mqtt_is_connected = False
    MLOpsStore.mlops_log_mqtt_lock.release()

    # logging.info("on_client_mqtt_disconnected: {}.".format(MLOpsStore.mlops_log_mqtt_is_connected))


def on_log_mqtt_connected(mqtt_client_object):
    if MLOpsStore.mlops_metrics is None:
        MLOpsStore.mlops_metrics = MLOpsMetrics()
        MLOpsStore.mlops_metrics.set_messenger(MLOpsStore.mlops_log_mqtt_mgr, MLOpsStore.mlops_args)

    MLOpsStore.mlops_metrics.run_id = MLOpsStore.mlops_run_id
    MLOpsStore.mlops_metrics.edge_id = MLOpsStore.mlops_edge_id

    if MLOpsStore.mlops_event is None:
        MLOpsStore.mlops_event = MLOpsProfilerEvent(MLOpsStore.mlops_args)
        MLOpsStore.mlops_event.set_messenger(MLOpsStore.mlops_log_mqtt_mgr, MLOpsStore.mlops_args)

    MLOpsStore.mlops_event.run_id = MLOpsStore.mlops_run_id
    MLOpsStore.mlops_event.edge_id = MLOpsStore.mlops_edge_id

    if MLOpsStore.mlops_log_mqtt_lock is None:
        MLOpsStore.mlops_log_mqtt_lock = threading.Lock()

    MLOpsStore.mlops_log_mqtt_lock.acquire()
    MLOpsStore.mlops_log_mqtt_is_connected = True
    MLOpsStore.mlops_log_mqtt_lock.release()

    # logging.info("on_client_mqtt_connected: {}.".format(MLOpsStore.mlops_log_mqtt_is_connected))


def setup_log_mqtt_mgr():
    if MLOpsStore.mlops_log_mqtt_mgr is not None:
        return

    if MLOpsStore.mlops_log_mqtt_lock is None:
        MLOpsStore.mlops_log_mqtt_lock = threading.Lock()

    if len(MLOpsStore.mlops_log_agent_config) == 0:
        return

    # logging.info(
    #    "mlops log metrics agent config: {},{}".format(MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_HOST"],
    #                                                   MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_PORT"]))

    if MLOpsStore.mlops_args is not None and hasattr(MLOpsStore.mlops_args, "device_id") and \
            MLOpsStore.mlops_args.device_id is not None:
        device_id = MLOpsStore.mlops_args.device_id
    else:
        device_id = str(uuid.uuid4())

    MLOpsStore.mlops_log_mqtt_mgr = MqttManager(
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_HOST"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_PORT"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["MQTT_USER"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["MQTT_PWD"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
        "FedML_MLOps_Metrics_{}_{}_{}".format(device_id,
                                              str(MLOpsStore.mlops_edge_id),
                                              str(uuid.uuid4()))
    )
    MLOpsStore.mlops_log_mqtt_mgr.add_connected_listener(on_log_mqtt_connected)
    MLOpsStore.mlops_log_mqtt_mgr.add_disconnected_listener(on_log_mqtt_disconnected)
    MLOpsStore.mlops_log_mqtt_mgr.connect()
    MLOpsStore.mlops_log_mqtt_mgr.loop_start()

    if MLOpsStore.mlops_metrics is None:
        MLOpsStore.mlops_metrics = MLOpsMetrics()
        MLOpsStore.mlops_metrics.set_messenger(MLOpsStore.mlops_log_mqtt_mgr, MLOpsStore.mlops_args)

    MLOpsStore.mlops_metrics.run_id = MLOpsStore.mlops_run_id
    MLOpsStore.mlops_metrics.edge_id = MLOpsStore.mlops_edge_id

    if MLOpsStore.mlops_event is None:
        MLOpsStore.mlops_event = MLOpsProfilerEvent(MLOpsStore.mlops_args)
        MLOpsStore.mlops_event.set_messenger(MLOpsStore.mlops_log_mqtt_mgr, MLOpsStore.mlops_args)

    MLOpsStore.mlops_event.run_id = MLOpsStore.mlops_run_id
    MLOpsStore.mlops_event.edge_id = MLOpsStore.mlops_edge_id


def release_log_mqtt_mgr():
    if MLOpsStore.mlops_log_mqtt_mgr is not None:
        MLOpsStore.mlops_log_mqtt_mgr.disconnect()
        MLOpsStore.mlops_log_mqtt_mgr.loop_stop()

    MLOpsStore.mlops_log_mqtt_lock.acquire()
    if MLOpsStore.mlops_log_mqtt_mgr is not None:
        MLOpsStore.mlops_log_mqtt_is_connected = False
    MLOpsStore.mlops_log_mqtt_lock.release()


def init_logs(args, edge_id):
    # Init runtime logs
    args.log_file_dir = ClientConstants.get_log_file_dir()
    args.run_id = MLOpsStore.mlops_run_id
    if hasattr(args, "rank") and args.rank is not None:
        if str(args.rank) == "0":
            args.role = "server"
        else:
            args.role = "client"
    else:
        args.role = "client"
    client_ids = list()
    client_ids.append(edge_id)
    args.client_id_list = json.dumps(client_ids)
    MLOpsRuntimeLog.get_instance(args).init_logs()

    # Start log processor for current run
    MLOpsRuntimeLogDaemon.get_instance(args).start_log_processor(MLOpsStore.mlops_run_id, MLOpsStore.mlops_edge_id)

    logging.info("client ids:{}".format(args.client_id_list))


def bind_simulation_device(args, userid):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ClientConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    version = fedml.get_env_version()
    setattr(args, "version", version)
    if args.rank == 0:
        setattr(args, "log_file_dir", ServerConstants.get_log_file_dir())
        setattr(args, "device_id",
                FedMLAccountManager.get_device_id(ServerConstants.get_data_dir()))
        runner = FedMLLaunchMasterProtocolManager(args)
    else:
        setattr(args, "log_file_dir", ClientConstants.get_log_file_dir())
        setattr(args, "device_id", FedMLAccountManager.get_device_id())
        runner = FedMLSlaveProtocolManager(args)
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")

    # Fetch configs from the MLOps config server.
    service_config = dict()
    config_try_count = 0
    edge_id = 0
    while config_try_count < 5:
        try:
            mqtt_config, s3_config, mlops_config, docker_config = runner.fetch_configs()
            service_config["mqtt_config"] = mqtt_config
            service_config["s3_config"] = s3_config
            service_config["ml_ops_config"] = mlops_config
            service_config["docker_config"] = docker_config
            runner.agent_config = service_config
            MLOpsStore.mlops_log_agent_config = service_config
            # setattr(args, "mqtt_config_path", mqtt_config)
            # setattr(args, "s3_config_path", s3_config)
            setattr(args, "log_server_url", mlops_config["LOG_SERVER_URL"])
            break
        except Exception as e:
            config_try_count += 1
            time.sleep(0.5)
            continue

    if config_try_count >= 5:
        logging.info("\nNote: Internet is not connected. "
                     "Experimental tracking results will not be synchronized to the MLOps (open.fedml.ai).\n")
        return False

    # Build unique device id
    if args.device_id is not None and len(str(args.device_id)) > 0:
        device_role = "Edge.Simulator"
        unique_device_id = "{}@{}.{}".format(args.device_id, args.os_name, device_role)

    # Bind account id to FedML® Nexus AI Platform
    register_try_count = 0
    edge_id = -1
    while register_try_count < 5:
        try:
            edge_id, _, _ = runner.bind_account_and_device_id(
                service_config["ml_ops_config"]["EDGE_BINDING_URL"],
                args.account_id, unique_device_id, args.os_name
            )
            if edge_id > 0:
                runner.edge_id = edge_id
                break
        except Exception as e:
            register_try_count += 1
            time.sleep(3)
            continue

    if edge_id <= 0:
        print("Oops, you failed to login the FedML MLOps platform.")
        print("Please check whether your network is normal!")
        return False
    MLOpsStore.mlops_edge_id = edge_id
    setattr(MLOpsStore.mlops_args, "client_id", edge_id)

    # Log arguments and binding results.
    runner.unique_device_id = unique_device_id

    MLOpsStore.mlops_args = args

    return True


def fetch_config(args, version="release"):
    setattr(args, "current_running_dir", ClientConstants.get_fedml_home_dir())
    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    if args.rank == 0:
        setattr(args, "log_file_dir", ServerConstants.get_log_file_dir())
        setattr(args, "device_id", FedMLAccountManager.get_device_id(ServerConstants.get_data_dir()))
    else:
        setattr(args, "log_file_dir", ClientConstants.get_log_file_dir())
        setattr(args, "device_id", FedMLAccountManager.get_device_id(ClientConstants.get_data_dir()))
    setattr(args, "config_version", version)
    setattr(args, "cloud_region", "")

    # Fetch configs from the MLOps config server.
    service_config = dict()
    config_try_count = 0
    edge_id = 0
    while config_try_count < 5:
        try:
            mqtt_config, s3_config, mlops_config, docker_config = MLOpsConfigs.fetch_all_configs()
            service_config["mqtt_config"] = mqtt_config
            service_config["s3_config"] = s3_config
            service_config["ml_ops_config"] = mlops_config
            service_config["docker_config"] = docker_config
            MLOpsStore.mlops_log_agent_config = service_config
            setattr(args, "mqtt_config_path", mqtt_config)
            setattr(args, "s3_config_path", s3_config)
            setattr(args, "log_server_url", mlops_config["LOG_SERVER_URL"])
            break
        except Exception as e:
            config_try_count += 1
            time.sleep(0.5)
            continue

    if config_try_count >= 5:
        logging.info("\nNote: Internet is not connected. "
                     "Experimental tracking results will not be synchronized to the MLOps (open.fedml.ai).\n")
        return False


def set_realtime_params():
    should_parse_args = False
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        return
    else:
        MLOpsStore.mlops_bind_result = True
        should_parse_args = True

    if should_parse_args:
        if MLOpsStore.mlops_args is not None:
            MLOpsStore.mlops_run_id = MLOpsStore.mlops_args.run_id
            if MLOpsStore.mlops_args.rank == 0:
                if hasattr(MLOpsStore.mlops_args, "server_id"):
                    MLOpsStore.mlops_edge_id = MLOpsStore.mlops_args.server_id
                else:
                    MLOpsStore.mlops_edge_id = 0
            else:
                if hasattr(MLOpsStore.mlops_args, "client_id"):
                    MLOpsStore.mlops_edge_id = MLOpsStore.mlops_args.client_id
                elif hasattr(MLOpsStore.mlops_args, "client_id_list"):
                    MLOpsStore.mlops_edge_id = json.loads(MLOpsStore.mlops_args.client_id_list)[0]
                else:
                    MLOpsStore.mlops_edge_id = 0

            if hasattr(MLOpsStore.mlops_args, "server_agent_id"):
                MLOpsStore.server_agent_id = MLOpsStore.mlops_args.server_agent_id
            else:
                MLOpsStore.server_agent_id = MLOpsStore.mlops_edge_id

    return True


def mlops_simulator_login(userid, run_id):
    # pass
    if not sys_utils.edge_simulator_has_login():
        subprocess.Popen(
            ["fedml", "login", str(userid),
             "-v", MLOpsStore.mlops_args.version,
             "-c", "-r", "edge_simulator"])

    sys_utils.save_simulator_process(ClientConstants.get_data_dir(),
                                     ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME, os.getpid(), str(run_id))


def mlops_parrot_enabled(args):
    if (
            hasattr(args, "enable_tracking")
            and args.enable_tracking is True
            and args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
    ):
        return True
    else:
        return False


def mlops_enabled(args):
    if args is None:
        MLOpsStore.mlops_args = get_fedml_args()
        args = MLOpsStore.mlops_args
    if hasattr(args, "using_mlops") and args.using_mlops:
        return True
    else:
        return False


def enable_logging_to_file(edge_id):
    args = get_fedml_args()
    # Init runtime logs
    args.log_file_dir = ""
    args.run_id = 0
    args.role = "client"
    client_ids = list()
    client_ids.append(edge_id)
    args.client_id_list = json.dumps(client_ids)
    setattr(args, "using_mlops", True)
    MLOpsRuntimeLog.get_instance(args).init_logs()
    return args


def release_resources(run_id, device_id):
    fedml_args = get_fedml_args()

    setup_log_mqtt_mgr()

    payload = {"run_id": run_id, "device_id": device_id, "gpu_count": 0}
    MLOpsStore.mlops_log_mqtt_mgr.send_message_json(
        MLOpsConstants.MSG_TOPIC_LAUNCH_RELEASE_GPU_IDS, json.dumps(payload))


def sync_deploy_id(device_id, master_deploy_id, worker_deploy_id_list, message_center=None):
    payload = {"device_id": device_id, "master_deploy_id": master_deploy_id, "worker_deploy_ids": worker_deploy_id_list}
    if message_center is None:
        fedml_args = get_fedml_args()
        setup_log_mqtt_mgr()
        MLOpsStore.mlops_log_mqtt_mgr.send_message_json(
            MLOpsConstants.MSG_TOPIC_LAUNCH_SYNC_DEPLOY_IDS, json.dumps(payload))
    else:
        message_center.send_message( MLOpsConstants.MSG_TOPIC_LAUNCH_SYNC_DEPLOY_IDS, json.dumps(payload))


