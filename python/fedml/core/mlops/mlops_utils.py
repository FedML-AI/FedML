import json
import multiprocessing
import os
import time
from dataclasses import dataclass, asdict
from os.path import expanduser
from typing import Dict, Any

import yaml


class MLOpsUtils:
    _ntp_offset = None
    BYTES_TO_GB = 1 / (1024 * 1024 * 1024)

    @staticmethod
    def calc_ntp_from_config(mlops_config):
        if mlops_config is None:
            return

        ntp_response = mlops_config.get("NTP_RESPONSE", None)
        if ntp_response is None:
            return

        # setup ntp time from the configs
        device_recv_time = int(time.time() * 1000)
        device_send_time = ntp_response.get("deviceSendTime", None)
        server_recv_time = ntp_response.get("serverRecvTime", None)
        server_send_time = ntp_response.get("serverSendTime", None)
        if device_send_time is None or server_recv_time is None or server_send_time is None:
            return

        # calculate the time offset(int)
        ntp_time = (server_recv_time + server_send_time + device_recv_time - device_send_time) // 2
        ntp_offset = ntp_time - device_recv_time

        # set the time offset
        MLOpsUtils.set_ntp_offset(ntp_offset)

    @staticmethod
    def set_ntp_offset(ntp_offset):
        MLOpsUtils._ntp_offset = ntp_offset

    @staticmethod
    def get_ntp_time():
        if MLOpsUtils._ntp_offset is not None:
            return int(time.time() * 1000) + MLOpsUtils._ntp_offset
        return int(time.time() * 1000)

    @staticmethod
    def get_ntp_offset():
        return MLOpsUtils._ntp_offset

    @staticmethod
    def write_log_trace(log_trace):
        log_trace_dir = os.path.join(expanduser("~"), "fedml_log")
        if not os.path.exists(log_trace_dir):
            os.makedirs(log_trace_dir, exist_ok=True)

        log_file_obj = open(os.path.join(log_trace_dir, "logs.txt"), "a")
        log_file_obj.write("{}\n".format(log_trace))
        log_file_obj.close()


@dataclass
class LogFile:
    file_path: str
    uploaded_file_index: int = 0
    upload_complete: bool = False


class MLOpsLoggingUtils:
    LOG_CONFIG_FILE = "log-config.yaml"
    _lock = multiprocessing.Lock()

    @staticmethod
    def acquire_lock(block=True):
        return MLOpsLoggingUtils._lock.acquire(block)

    @staticmethod
    def release_lock():
        # Purposefully acquire lock with non-blocking call to make it idempotent
        MLOpsLoggingUtils._lock.acquire(block=False)
        MLOpsLoggingUtils._lock.release()

    @staticmethod
    def build_log_file_path_with_run_params(
            run_id, edge_id, log_file_dir, is_server=False, log_file_prefix=None
    ):
        program_prefix = "FedML-{} @device-id-{}".format(
            "Server" if is_server else "Client", edge_id)
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(
            log_file_dir, "fedml-run{}-{}-edge-{}.log".format(
                "" if log_file_prefix is None else f"-{log_file_prefix}", run_id, edge_id
            ))

        return log_file_path, program_prefix

    @staticmethod
    def build_log_file_path(args):
        edge_id = MLOpsLoggingUtils.get_edge_id_from_args(args)
        program_prefix = MLOpsLoggingUtils.get_program_prefix(args, edge_id)

        if not os.path.exists(args.log_file_dir):
            os.makedirs(args.log_file_dir, exist_ok=True)

        if hasattr(args, "log_file_path") and args.log_file_path is not None and len(args.log_file_path) > 0:
            log_file_path = args.log_file_path
        else:
            log_file_path = os.path.join(args.log_file_dir, "fedml-run-"
                                         + str(args.run_id)
                                         + "-edge-"
                                         + str(edge_id)
                                         + ".log")

        return log_file_path, program_prefix

    @staticmethod
    def get_program_prefix(args, edge_id):
        if args.role == "server":
            program_prefix = "FedML-Server @device-id-{edge}".format(edge=edge_id)
        else:
            program_prefix = "FedML-Client @device-id-{edge}".format(edge=edge_id)
        return program_prefix

    @staticmethod
    def get_edge_id_from_args(args):
        if args.role == "server":
            # Considering that 0 is a valid value, we need to ensure it is not None rather than solely checking
            # for truthiness
            if getattr(args, "server_id", None) is not None:
                edge_id = args.server_id
            else:
                if getattr(args, "edge_id", None) is not None:
                    edge_id = args.edge_id
                else:
                    edge_id = 0
        else:
            if getattr(args, "client_id", None) is not None:
                edge_id = args.client_id
            elif hasattr(args, "client_id_list"):
                if args.client_id_list is None:
                    edge_id = 0
                else:
                    edge_ids = json.loads(args.client_id_list)
                    if len(edge_ids) > 0:
                        edge_id = edge_ids[0]
                    else:
                        edge_id = 0
            else:
                if getattr(args, "edge_id", None) is not None:
                    edge_id = args.edge_id
                else:
                    edge_id = 0

        return edge_id

    @staticmethod
    def load_log_config(run_id, device_id, log_config_file) -> Dict[int, LogFile]:
        try:
            log_config_key = "log_config_{}_{}".format(run_id, device_id)
            log_config = MLOpsLoggingUtils.load_yaml_config(log_config_file)
            run_log_config = log_config.get(log_config_key, {})
            config_data = {}
            for index, data in run_log_config.items():
                config_data[index] = LogFile(**data)
            return config_data
        except Exception as e:
            raise ValueError("Error loading log config: {}".format(e))

    @staticmethod
    def save_log_config(run_id, device_id, log_config_file, config_data):
        try:
            log_config_key = "log_config_{}_{}".format(run_id, device_id)
            log_config = MLOpsLoggingUtils.load_yaml_config(log_config_file)
            log_config[log_config_key] = MLOpsLoggingUtils.__convert_to_dict(config_data)
            with open(log_config_file, "w") as stream:
                yaml.dump(log_config, stream)
        except Exception as e:
            MLOpsLoggingUtils.release_lock()
            raise ValueError("Error saving log config: {}".format(e))

    @staticmethod
    def load_yaml_config(log_config_file):
        """Helper function to load a yaml config file"""
        if MLOpsLoggingUtils._lock.acquire(block=False):
            MLOpsLoggingUtils._lock.release()
            raise ValueError("Able to acquire lock. This means lock was not acquired by the caller")
        if not os.path.exists(log_config_file):
            MLOpsLoggingUtils.generate_yaml_doc({}, log_config_file)
        with open(log_config_file, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise ValueError(f"Yaml error - check yaml file, error: {e}")

    @staticmethod
    def generate_yaml_doc(log_config_object, yaml_file):
        try:
            file = open(yaml_file, "w", encoding="utf-8")
            yaml.dump(log_config_object, file)
            file.close()
        except Exception as e:
            raise ValueError(f"Error generating yaml doc: {e}")

    @staticmethod
    def __convert_to_dict(obj: Any) -> Any:
        if isinstance(obj, LogFile):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: MLOpsLoggingUtils.__convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MLOpsLoggingUtils.__convert_to_dict(item) for item in obj]
        else:
            return obj
