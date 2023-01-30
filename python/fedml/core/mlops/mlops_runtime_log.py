import argparse
import json
import logging
import os
import sys
import threading
import time
from logging import handlers

from fedml import mlops


class MLOpsRuntimeLog:
    FED_LOG_LINE_NUMS_PER_UPLOADING = 1000
    FED_LOG_UPLOAD_FREQUENCY = 1
    FEDML_LOG_REPORTING_STATUS_FILE_NAME = "log_status.id"

    _log_sdk_instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(MLOpsRuntimeLog, "_instance"):
            with MLOpsRuntimeLog._instance_lock:
                if not hasattr(MLOpsRuntimeLog, "_instance"):
                    MLOpsRuntimeLog._instance = object.__new__(cls)
        return MLOpsRuntimeLog._instance

    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        if MLOpsRuntimeLog._log_sdk_instance is not None and \
                hasattr(MLOpsRuntimeLog._log_sdk_instance, "args") and \
                hasattr(MLOpsRuntimeLog._log_sdk_instance.args, "rank"):
            if MLOpsRuntimeLog._log_sdk_instance.args.rank == 0:
                mlops.log_aggregation_failed_status()
            else:
                mlops.log_training_failed_status()
        else:
            mlops.log_aggregation_failed_status()

    def __init__(self, args):
        self.logger = None
        self.args = args
        if hasattr(args, "using_mlops"):
            self.should_write_log_file = args.using_mlops
        else:
            self.should_write_log_file = False
        self.log_file_dir = args.log_file_dir
        self.log_file = None
        self.run_id = args.run_id
        if args.rank == 0:
            if hasattr(args, "server_id"):
                self.edge_id = args.server_id
            else:
                if hasattr(args, "edge_id"):
                    self.edge_id = args.edge_id
                else:
                    self.edge_id = 0
        else:
            if hasattr(args, "client_id"):
                self.edge_id = args.client_id
            elif hasattr(args, "client_id_list"):
                if args.client_id_list is None:
                    self.edge_id = 0
                else:
                    edge_ids = json.loads(args.client_id_list)
                    if len(edge_ids) > 0:
                        self.edge_id = edge_ids[0]
                    else:
                        self.edge_id = 0
            else:
                if hasattr(args, "edge_id"):
                    self.edge_id = args.edge_id
                else:
                    self.edge_id = 0

        self.origin_log_file_path = os.path.join(self.log_file_dir, "fedml-run-"
                                                 + str(self.run_id)
                                                 + "-edge-"
                                                 + str(self.edge_id)
                                                 + ".log")

        sys.excepthook = MLOpsRuntimeLog.handle_exception

    @staticmethod
    def get_instance(args):
        if MLOpsRuntimeLog._log_sdk_instance is None:
            MLOpsRuntimeLog._log_sdk_instance = MLOpsRuntimeLog(args)

        return MLOpsRuntimeLog._log_sdk_instance

    def init_logs(self, show_stdout_log=True):
        log_file_path, program_prefix = MLOpsRuntimeLog.build_log_file_path(self.args)
        logging.raiseExceptions = True
        self.logger = logging.getLogger(log_file_path)
        format_str = logging.Formatter(fmt="[" + program_prefix + "] [%(asctime)s] [%(levelname)s] "
                                                                  "[%(filename)s:%(lineno)d:%(funcName)s] %("
                                                                  "message)s",
                                       datefmt="%a, %d %b %Y %H:%M:%S")
        stdout_handle = logging.StreamHandler()
        stdout_handle.setFormatter(format_str)
        if show_stdout_log:
            stdout_handle.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)
        else:
            stdout_handle.setLevel(logging.CRITICAL)
            self.logger.setLevel(logging.CRITICAL)
        self.logger.addHandler(stdout_handle)
        if hasattr(self, "should_write_log_file") and self.should_write_log_file:
            when = 'D'
            backup_count = 100
            file_handle = handlers.TimedRotatingFileHandler(filename=log_file_path, when=when,
                                                            backupCount=backup_count, encoding='utf-8')
            file_handle.setFormatter(format_str)
            file_handle.setLevel(logging.INFO)
            self.logger.addHandler(file_handle)
        logging.root = self.logger

        import sys

        class GlobalLogger(object):
            def __init__(self, filename='default.log', stream=sys.stdout, format_str=""):
                self.terminal = stream
                self.log = open(filename, 'a')
                self.format_str = format_str
                if self.log is not None:
                    print("open log file")

            def write(self, message):
                self.terminal.write("{}{}".format(format_str, message))
                self.log.write("{}{}".format(format_str, message))

            def flush(self):
                pass

        #sys.stdout = GlobalLogger(log_file_path, sys.stdout)
        #sys.stderr = GlobalLogger(log_file_path, sys.stderr)
        #print("Make `print` output to log files.", file=GlobalLogger(log_file_path, sys.stdout).log)

    @staticmethod
    def build_log_file_path(in_args):
        if in_args.rank == 0:
            if hasattr(in_args, "server_id"):
                edge_id = in_args.server_id
            else:
                if hasattr(in_args, "edge_id"):
                    edge_id = in_args.edge_id
                else:
                    edge_id = 0
            program_prefix = "FedML-Server({}) @device-id-{}".format(in_args.rank, edge_id)
        else:
            if hasattr(in_args, "client_id"):
                edge_id = in_args.client_id
            elif hasattr(in_args, "client_id_list"):
                if in_args.client_id_list is None:
                    edge_id = 0
                else:
                    edge_ids = json.loads(in_args.client_id_list)
                    if len(edge_ids) > 0:
                        edge_id = edge_ids[0]
                    else:
                        edge_id = 0
            else:
                if hasattr(in_args, "edge_id"):
                    edge_id = in_args.edge_id
                else:
                    edge_id = 0
            program_prefix = "FedML-Client({rank}) @device-id-{edge}".format(
                rank=in_args.rank, edge=edge_id
            )

        os.system("mkdir -p " + in_args.log_file_dir)
        log_file_path = os.path.join(in_args.log_file_dir, "fedml-run-"
                                     + str(in_args.run_id)
                                     + "-edge-"
                                     + str(edge_id)
                                     + ".log")

        return log_file_path, program_prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_file_dir", "-log", help="log file dir")
    parser.add_argument("--run_id", "-ri", type=str,
                        help='run id')
    parser.add_argument("--rank", "-r", type=str, default="1")
    parser.add_argument("--server_id", "-s", type=str, default="1")
    parser.add_argument("--client_id", "-c", type=str, default="1")
    parser.add_argument("--client_id_list", "-cil", type=str, default="[]")

    args = parser.parse_args()
    setattr(args, "using_mlops", True)
    setattr(args, "config_version", "local")
    MLOpsRuntimeLog.get_instance(args).init_logs()

    count = 0
    while True:
        logging.info("Test Log {}".format(count))
        count += 1
        time.sleep(1)
