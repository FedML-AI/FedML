import json
import logging
import os
import platform
import subprocess
import threading
import time
import uuid

import click
import requests
from fedml.cli.comm_utils import sys_utils
from fedml.core.mlops.mlops_configs import MLOpsConfigs

from ...cli.edge_deployment.client_constants import ClientConstants
from ...cli.edge_deployment.client_runner import FedMLClientRunner
from ...cli.server_deployment.server_runner import FedMLServerRunner
from ...constants import FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_TRAINING_PLATFORM_SIMULATION_TYPE
from ...cli.server_deployment.server_constants import ServerConstants

from ..distributed.communication.mqtt.mqtt_manager import MqttManager
from ..distributed.communication.s3.remote_storage import S3Storage

from .mlops_metrics import MLOpsMetrics
from .mlops_profiler_event import MLOpsProfilerEvent
from .system_stats import SysStats
from .mlops_status import MLOpsStatus
from .mlops_runtime_log import MLOpsRuntimeLog
from .mlops_runtime_log_daemon import MLOpsRuntimeLogProcessor
from .mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ...cli.edge_deployment.client_data_interface import FedMLClientDataInterface

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
    "log_training_failed_status"
]


class MLOpsStore:
    mlops_args = None
    mlops_project_id: int = None
    mlops_run_id = None
    mlops_edge_id = None
    mlops_log_metrics = dict()
    mlops_log_round_info = dict()
    mlops_log_client_training_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING
    mlops_log_server_training_status = ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING
    mlops_log_round_start_time = 0.0
    mlops_log_metrics_lock = None
    mlops_log_mqtt_mgr = None
    mlops_log_mqtt_lock = None
    mlops_log_mqtt_is_connected = False
    mlops_log_agent_config = dict()
    mlops_metrics = None
    mlops_event = None
    mlops_bind_result = False
    server_agent_id = None
    current_parrot_process = None

    def __init__(self):
        pass


def pre_setup(args):
    MLOpsStore.mlops_args = args


def init(args):
    MLOpsStore.mlops_args = args
    if not mlops_parrot_enabled(args):
        if not hasattr(args, "config_version"):
            args.config_version = "release"
        fetch_config(args, args.config_version)
        MLOpsRuntimeLog.get_instance(args).init_logs()
        return
    else:
        if hasattr(args, "simulator_daemon"):
            # Bind local device as simulation device on the MLOps platform.
            setattr(args, "using_mlops", True)
            setattr(args, "rank", 1)
            MLOpsStore.mlops_bind_result = bind_simulation_device(args, args.user, args.version)
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

    # Bind local device as simulation device on the MLOps platform.
    setattr(args, "using_mlops", True)
    setattr(args, "rank", 1)
    MLOpsStore.mlops_bind_result = bind_simulation_device(args, api_key, args.config_version)
    if not MLOpsStore.mlops_bind_result:
        setattr(args, "using_mlops", False)
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
        click.echo("Failed to init project and run.")
        return

    # Init runtime logs
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
    wait_log_mqtt_connected()

    if event_started:
        MLOpsStore.mlops_event.log_event_started(event_name, event_value, event_edge_id)
    else:
        MLOpsStore.mlops_event.log_event_ended(event_name, event_value, event_edge_id)


def log(metrics: dict, commit=True):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    if MLOpsStore.mlops_log_metrics_lock is None:
        MLOpsStore.mlops_log_metrics_lock = threading.Lock()

    MLOpsStore.mlops_log_metrics_lock.acquire()
    for k, v in metrics.items():
        k = str(k).replace("/", "_")
        if k.startswith("round"):
            k = "round_idx"

        # if isinstance(v, int):
        #     # k = "round_idx"
        #     k = "round_idx_" + k
        MLOpsStore.mlops_log_metrics[k] = v
    MLOpsStore.mlops_log_metrics["run_id"] = str(MLOpsStore.mlops_run_id)
    MLOpsStore.mlops_log_metrics["timestamp"] = float(time.time_ns() / 1000 / 1000 * 1.0)
    MLOpsStore.mlops_log_metrics_lock.release()

    logging.info("log metrics {}".format(json.dumps(MLOpsStore.mlops_log_metrics)))

    if commit:
        setup_log_mqtt_mgr()
        wait_log_mqtt_connected()
        MLOpsStore.mlops_log_metrics_lock.acquire()
        MLOpsStore.mlops_metrics.report_server_training_metric(MLOpsStore.mlops_log_metrics)
        MLOpsStore.mlops_log_metrics.clear()
        MLOpsStore.mlops_log_metrics_lock.release()
        release_log_mqtt_mgr()


def log_training_status(status, run_id=None):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    logging.info("log training status {}".format(status))

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        MLOpsStore.mlops_metrics.broadcast_client_training_status(MLOpsStore.mlops_edge_id, status)
    else:
        MLOpsStore.mlops_metrics.report_client_training_status(MLOpsStore.mlops_edge_id, status)
    release_log_mqtt_mgr()


def log_aggregation_status(status, run_id=None):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    if run_id is not None:
        MLOpsStore.mlops_args.run_id = run_id
        MLOpsStore.mlops_run_id = run_id
    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    logging.info("log aggregation status {}".format(status))

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        device_role = "simulator"
    else:
        device_role = "server"
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        MLOpsStore.mlops_metrics.broadcast_server_training_status(MLOpsStore.mlops_run_id, status, role=device_role)
        sys_utils.save_simulator_process(ClientConstants.get_data_dir(),
                                         ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME, os.getpid(),
                                         str(MLOpsStore.mlops_run_id),
                                         run_status=status)

        # Start log processor for current run
        if status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
            MLOpsRuntimeLogDaemon.get_instance(MLOpsStore.mlops_args).stop_log_processor(MLOpsStore.mlops_run_id,
                                                                                         MLOpsStore.mlops_edge_id)
    else:
        MLOpsStore.mlops_metrics.report_server_training_status(MLOpsStore.mlops_run_id, status, role=device_role)
    release_log_mqtt_mgr()


def log_training_finished_status(run_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_training_status(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED, run_id)
        time.sleep(2)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    logging.info("log training inner status {}".format(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED))

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    MLOpsStore.mlops_metrics.broadcast_client_training_status(MLOpsStore.mlops_edge_id,
                                                              ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED)
    MLOpsStore.mlops_metrics.report_client_id_status(MLOpsStore.mlops_run_id,
                                                     MLOpsStore.mlops_edge_id,
                                                     ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED)
    release_log_mqtt_mgr()


def send_exit_train_msg(run_id=None):
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    MLOpsStore.mlops_metrics.client_send_exit_train_msg()
    release_log_mqtt_mgr()


def log_training_failed_status(run_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_training_status(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED, run_id)
        time.sleep(2)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    logging.info("log training inner status {}".format(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED))

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    MLOpsStore.mlops_metrics.broadcast_client_training_status(MLOpsStore.mlops_edge_id,
                                                              ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
    MLOpsStore.mlops_metrics.report_client_id_status(MLOpsStore.mlops_run_id,
                                                     MLOpsStore.mlops_edge_id,
                                                     ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
    release_log_mqtt_mgr()


def log_aggregation_finished_status(run_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_aggregation_status(ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED, run_id)
        time.sleep(15)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    logging.info("log aggregation inner status {}".format(ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED))

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    MLOpsStore.mlops_metrics.broadcast_server_training_status(MLOpsStore.mlops_run_id,
                                                              ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED)
    MLOpsStore.mlops_metrics.report_server_id_status(MLOpsStore.mlops_run_id,
                                                     ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED)
    release_log_mqtt_mgr()


def log_aggregation_failed_status(run_id=None):
    if mlops_parrot_enabled(MLOpsStore.mlops_args):
        log_aggregation_status(ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, run_id)
        return

    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    # logging.info("log aggregation inner status {}".format(ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED))

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    MLOpsStore.mlops_metrics.broadcast_server_training_status(MLOpsStore.mlops_run_id,
                                                              ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)
    MLOpsStore.mlops_metrics.report_server_id_status(MLOpsStore.mlops_run_id,
                                                     ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)
    release_log_mqtt_mgr()


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
    wait_log_mqtt_connected()
    model_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "round_idx": round_index,
        "global_aggregated_model_s3_address": model_url,
    }
    MLOpsStore.mlops_metrics.report_aggregated_model_info(model_info)
    release_log_mqtt_mgr()


def log_training_model_net_info(model_net):
    if model_net is None:
        return
    if not mlops_enabled(MLOpsStore.mlops_args):
        return

    set_realtime_params()

    if not MLOpsStore.mlops_bind_result:
        return

    s3_config = MLOpsStore.mlops_log_agent_config.get("s3_config", None)
    if s3_config is None:
        return
    s3_client = S3Storage(s3_config)
    model_key = "fedml-model-net-run-{}-{}".format(str(MLOpsStore.mlops_run_id), str(uuid.uuid4()))
    model_url = s3_client.write_model_net(model_key, model_net, ClientConstants.get_model_cache_dir())

    logging.info("log training model net info {}".format(model_url))

    setup_log_mqtt_mgr()
    wait_log_mqtt_connected()
    model_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "training_model_net_s3_address": model_url,
    }
    MLOpsStore.mlops_metrics.report_training_model_net_info(model_info)
    release_log_mqtt_mgr()


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
    wait_log_mqtt_connected()
    model_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "edge_id": MLOpsStore.mlops_edge_id,
        "round_idx": round_index,
        "client_model_s3_address": model_url,
    }
    MLOpsStore.mlops_metrics.report_client_model_info(model_info)
    release_log_mqtt_mgr()

    FedMLClientDataInterface.get_instance().save_running_job(MLOpsStore.mlops_run_id, MLOpsStore.mlops_edge_id,
                                                             round_index,
                                                             total_rounds,
                                                             "Running")


def log_sys_perf(sys_args=None):
    if not mlops_enabled(sys_args):
        return

    if sys_args is not None:
        MLOpsStore.mlops_args = sys_args

    MLOpsMetrics.report_sys_perf(MLOpsStore.mlops_args)


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
    wait_log_mqtt_connected()
    round_info = {
        "run_id": MLOpsStore.mlops_run_id,
        "round_index": round_index,
        "total_rounds": total_rounds,
        "running_time": round(time.time() - MLOpsStore.mlops_log_round_start_time, 4),
    }
    logging.info("log round info {}".format(round_info))
    MLOpsStore.mlops_metrics.report_server_training_round_info(round_info)
    release_log_mqtt_mgr()


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
    url = "https://open.fedml.ai"
    config_version = "release"
    if (
            hasattr(args, "config_version")
            and args.config_version is not None
    ):
        # Setup config url based on selected version.
        config_version = args.config_version
        if args.config_version == "release":
            url = "https://open.fedml.ai"
        elif args.config_version == "test":
            url = "https://open-test.fedml.ai"
        elif args.config_version == "dev":
            url = "https://open-dev.fedml.ai"
        elif args.config_version == "local":
            if hasattr(args, "local_server") and args.local_server is not None:
                url = "http://{}:9000".format(args.local_server)
            else:
                url = "http://localhost:9000"

    cert_path = None
    if str(url).startswith("https://"):
        cur_source_dir = os.path.dirname(__file__)
        cert_path = os.path.join(
            cur_source_dir, "ssl", "open-" + config_version + ".fedml.ai_bundle.crt"
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

    if MLOpsStore.mlops_log_mqtt_mgr is not None:
        MLOpsStore.mlops_log_mqtt_lock.acquire()
        MLOpsStore.mlops_log_mqtt_mgr.remove_disconnected_listener(on_log_mqtt_disconnected)
        MLOpsStore.mlops_log_mqtt_is_connected = False
        MLOpsStore.mlops_log_mqtt_mgr.disconnect()
        MLOpsStore.mlops_log_mqtt_mgr = None
        MLOpsStore.mlops_log_mqtt_lock.release()

    if len(MLOpsStore.mlops_log_agent_config) == 0:
        return

    # logging.info(
    #    "mlops log metrics agent config: {},{}".format(MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_HOST"],
    #                                                   MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_PORT"]))

    MLOpsStore.mlops_log_mqtt_mgr = MqttManager(
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_HOST"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["BROKER_PORT"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["MQTT_USER"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["MQTT_PWD"],
        MLOpsStore.mlops_log_agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
        "FedML_MLOps_Metrics_" + MLOpsStore.mlops_args.device_id + "_" + str(MLOpsStore.mlops_edge_id)
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


def release_log_mqtt_mgr(real_release=False):
    if real_release:
        if MLOpsStore.mlops_log_mqtt_mgr is not None:
            MLOpsStore.mlops_log_mqtt_mgr.disconnect()
            MLOpsStore.mlops_log_mqtt_mgr.loop_stop()

        MLOpsStore.mlops_log_mqtt_lock.acquire()
        if MLOpsStore.mlops_log_mqtt_mgr is not None:
            MLOpsStore.mlops_log_mqtt_is_connected = False
        MLOpsStore.mlops_log_mqtt_lock.release()


def wait_log_mqtt_connected():
    pass
    # while True:
    #     MLOpsStore.mlops_log_mqtt_lock.acquire()
    #     if MLOpsStore.mlops_log_mqtt_is_connected is True \
    #             and MLOpsStore.mlops_metrics is not None:
    #         MLOpsStore.mlops_metrics.set_messenger(MLOpsStore.mlops_log_mqtt_mgr, MLOpsStore.mlops_args)
    #         if MLOpsStore.mlops_event is not None:
    #             MLOpsStore.mlops_event.set_messenger(MLOpsStore.mlops_log_mqtt_mgr, MLOpsStore.mlops_args)
    #         MLOpsStore.mlops_log_mqtt_lock.release()
    #         break
    #     MLOpsStore.mlops_log_mqtt_lock.release()
    #     time.sleep(0.01)


def init_logs(args, edge_id):
    # Init runtime logs
    args.log_file_dir = ClientConstants.get_log_file_dir()
    args.run_id = MLOpsStore.mlops_run_id
    args.rank = 1
    client_ids = list()
    client_ids.append(edge_id)
    args.client_id_list = json.dumps(client_ids)
    MLOpsRuntimeLog.get_instance(args).init_logs()

    # Start log processor for current run
    MLOpsRuntimeLogDaemon.get_instance(args).start_log_processor(MLOpsStore.mlops_run_id, MLOpsStore.mlops_edge_id)

    logging.info("client ids:{}".format(args.client_id_list))


def bind_simulation_device(args, userid, version="release"):
    setattr(args, "account_id", userid)
    setattr(args, "current_running_dir", ClientConstants.get_fedml_home_dir())

    sys_name = platform.system()
    if sys_name == "Darwin":
        sys_name = "MacOS"
    setattr(args, "os_name", sys_name)
    setattr(args, "version", version)
    if args.rank == 0:
        setattr(args, "log_file_dir", ServerConstants.get_log_file_dir())
        setattr(args, "device_id", FedMLServerRunner.get_device_id())
        runner = FedMLServerRunner(args)
    else:
        setattr(args, "log_file_dir", ClientConstants.get_log_file_dir())
        setattr(args, "device_id", FedMLClientRunner.get_device_id())
        runner = FedMLClientRunner(args)
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
        click.echo("\nNote: Internet is not connected. "
                   "Experimental tracking results will not be synchronized to the MLOps (open.fedml.ai).\n")
        return False

    # Build unique device id
    if args.device_id is not None and len(str(args.device_id)) > 0:
        device_role = "Edge.Simulator"
        unique_device_id = "{}@{}.{}".format(args.device_id, args.os_name, device_role)

    # Bind account id to the MLOps platform.
    register_try_count = 0
    edge_id = 0
    while register_try_count < 5:
        try:
            edge_id = runner.bind_account_and_device_id(
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
        click.echo("Oops, you failed to login the FedML MLOps platform.")
        click.echo("Please check whether your network is normal!")
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
        setattr(args, "device_id", FedMLServerRunner.get_device_id())
        runner = FedMLServerRunner(args)
    else:
        setattr(args, "log_file_dir", ClientConstants.get_log_file_dir())
        setattr(args, "device_id", FedMLClientRunner.get_device_id())
        runner = FedMLClientRunner(args)
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
            setattr(args, "mqtt_config_path", mqtt_config)
            setattr(args, "s3_config_path", s3_config)
            setattr(args, "log_server_url", mlops_config["LOG_SERVER_URL"])
            break
        except Exception as e:
            config_try_count += 1
            time.sleep(0.5)
            continue

    if config_try_count >= 5:
        click.echo("\nNote: Internet is not connected. "
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
    if hasattr(args, "using_mlops") and args.using_mlops:
        return True
    else:
        return False
