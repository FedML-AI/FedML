import argparse
import datetime
import logging
import os
import sys
import threading
import time
import shutil
from logging.handlers import TimedRotatingFileHandler

from fedml import mlops
from fedml.core.mlops.mlops_utils import MLOpsUtils, MLOpsLoggingUtils, LogFile

LOG_LEVEL = logging.INFO
ROTATION_FREQUENCY = 'D'
# when rollover is done, no more than backupCount files are kept - the oldest ones are deleted.
BACKUP_COUNT = 100


class MLOpsFileHandler(TimedRotatingFileHandler):

    def __init__(self, run_id, edge_id, log_config_file, filepath):
        super().__init__(filename=filepath, when=ROTATION_FREQUENCY,
                         backupCount=BACKUP_COUNT,encoding='utf-8')
        self.run_id = run_id
        self.edge_id = edge_id
        self.file_path = filepath
        self.rotate_count = 0
        self.rotator: callable = self.update_config_and_rotate
        self.log_config_file = log_config_file
        self.__initialize_config()

    def update_config_and_rotate(self, source, dest):
        # source = current log file name
        # dest = log file name (dated)
        MLOpsLoggingUtils.acquire_lock()

        # Check if the source and destination files exist. If it does, return
        if os.path.exists(source):
            # Copy the contents of the source file to the destination file
            shutil.copy(source, dest)
            # Clear everything in the source file
            with open(source, 'w') as src_file:
                src_file.truncate(0)
            src_file.close()

        config_data = MLOpsLoggingUtils.load_log_config(self.run_id, self.edge_id,
                                                        self.log_config_file)

        # Update file name of current log file
        config_data[self.rotate_count].file_path = dest
        self.rotate_count += 1

        # Store the rotate count, and corresponding log file name in the config file
        rotated_log_file = LogFile(file_path=source)
        config_data[self.rotate_count] = rotated_log_file
        MLOpsLoggingUtils.save_log_config(run_id=self.run_id, device_id=self.edge_id,
                                          log_config_file=self.log_config_file,
                                          config_data=config_data)
        MLOpsLoggingUtils.release_lock()

    def __initialize_config(self):
        try:
            MLOpsLoggingUtils.acquire_lock()
            config_data = MLOpsLoggingUtils.load_log_config(run_id=self.run_id, device_id=self.edge_id,
                                                            log_config_file=self.log_config_file)
            if not config_data:
                log_file = LogFile(file_path=self.file_path)
                config_data = {self.rotate_count: log_file}
                MLOpsLoggingUtils.save_log_config(run_id=self.run_id, device_id=self.edge_id,
                                                  log_config_file=self.log_config_file, config_data=config_data)
        except Exception as e:
            raise ValueError("Error initializing log config: {}".format(e))
        finally:
            MLOpsLoggingUtils.release_lock()


class MLOpsFormatter(logging.Formatter):
    converter = datetime.datetime.utcfromtimestamp

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.ntp_offset = 0.0

    # Here the `record` is a LogRecord object
    def formatTime(self, record, datefmt=None):
        log_time = record.created
        if record.created is None:
            log_time = time.time()

        if self.ntp_offset is None:
            self.ntp_offset = 0.0

        log_ntp_time = int((log_time * 1000 + self.ntp_offset) / 1000.0)
        ct = self.converter(log_ntp_time)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%a, %d %b %Y %H:%M:%S")
            s = "%s.%09d" % (t, int((record.created % 1) * 1e9))  # Get nanoseconds from record.created
        return s


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
                mlops.log_aggregation_exception_status()
            else:
                mlops.log_training_failed_status()
        else:
            mlops.log_aggregation_exception_status()

    def __init__(self, args):
        self.format_str = None
        self.stdout_handle = None
        self.logger = None
        self.args = args
        if hasattr(args, "using_mlops"):
            self.should_write_log_file = args.using_mlops
        else:
            self.should_write_log_file = False
        if not hasattr(args, "log_file_dir"):
            setattr(args, "log_file_dir", "./logs")
        self.log_file_dir = args.log_file_dir
        self.log_file = None
        self.run_id = args.run_id
        self.edge_id = MLOpsLoggingUtils.get_edge_id_from_args(args)
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

    def init_logs(self, log_level=None):
        log_file_path, program_prefix = MLOpsLoggingUtils.build_log_file_path(self.args)
        logging.raiseExceptions = True
        self.logger = logging.getLogger(log_file_path)
        self.generate_format_str()
        self.stdout_handle = logging.StreamHandler()
        self.stdout_handle.setFormatter(self.format_str)
        log_level = log_level if log_level is not None else LOG_LEVEL
        self.stdout_handle.setLevel(log_level)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()
        self.logger.addHandler(self.stdout_handle)
        if hasattr(self, "should_write_log_file") and self.should_write_log_file:
            run_id, edge_id = self.args.run_id, MLOpsLoggingUtils.get_edge_id_from_args(self.args)
            log_config_file = os.path.join(self.log_file_dir, MLOpsLoggingUtils.LOG_CONFIG_FILE)
            file_handle = MLOpsFileHandler(filepath=log_file_path, log_config_file=log_config_file, run_id=run_id,
                                           edge_id=edge_id)
            file_handle.setFormatter(self.format_str)
            file_handle.setLevel(logging.INFO)
            self.logger.addHandler(file_handle)
        logging.root = self.logger
        # Rewrite sys.stdout to redirect stdout (i.e print()) to Logger
        sys.stdout.write = self.logger.info

    def enable_show_log_to_stdout(self, enable=True):
        if self.stdout_handle is None:
            return

        if enable:
            self.stdout_handle.setLevel(logging.INFO)
            self.logger.setLevel(logging.INFO)
            self.stdout_handle.setFormatter(None)
        else:
            self.stdout_handle.setLevel(logging.CRITICAL)
            self.logger.setLevel(logging.CRITICAL)
            if self.format_str is None:
                self.generate_format_str()
            self.stdout_handle.setFormatter(self.format_str)

    def generate_format_str(self):
        log_file_path, program_prefix = MLOpsLoggingUtils.build_log_file_path(self.args)
        self.format_str = MLOpsFormatter(fmt="[" + program_prefix + "] [%(asctime)s] [%(levelname)s] "
                                                                    "[%(filename)s:%(lineno)d:%(funcName)s] %("
                                                                    "message)s")
        self.format_str.ntp_offset = MLOpsUtils.get_ntp_offset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_file_dir", "-log", help="log file dir")
    parser.add_argument("--run_id", "-ri", type=str,
                        help='run id')
    parser.add_argument("--rank", "-r", type=str, default="1")
    parser.add_argument("--server_id", "-s", type=str, default="1")
    parser.add_argument("--client_id", "-c", type=str, default="1")
    parser.add_argument("--client_id_list", "-cil", type=str, default="[]")
    parser.add_argument("--role", "-role", type=str, default="client")

    args = parser.parse_args()
    setattr(args, "using_mlops", True)
    setattr(args, "config_version", "local")

    MLOpsRuntimeLog.get_instance(args).init_logs()

    count = 0
    while True:
        logging.info("Test Log {}".format(count))
        count += 1
        time.sleep(1)
