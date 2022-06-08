import json
import logging
import multiprocessing
import os
import shutil
import threading
import time

import requests
import yaml
from fedml.core.mlops.mlops_configs import MLOpsConfigs


class MLOpsRuntimeLog:
    FED_LOG_LINE_NUMS_PER_UPLOADING = 100
    FED_LOG_UPLOAD_FREQUENCY = 3

    _log_sdk_instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(MLOpsRuntimeLog, "_instance"):
            with MLOpsRuntimeLog._instance_lock:
                if not hasattr(MLOpsRuntimeLog, "_instance"):
                    MLOpsRuntimeLog._instance = object.__new__(cls)
        return MLOpsRuntimeLog._instance

    def __init__(self, args):
        self.args = args
        if hasattr(args, "using_mlops"):
            self.should_write_log_file = args.using_mlops
            self.should_upload_log_file = args.using_mlops
        else:
            self.should_write_log_file = False
            self.should_upload_log_file = False
        self.log_file_dir = args.log_file_dir
        self.log_file = None
        self.run_id = args.run_id
        if args.rank == 0:
            self.edge_id = 0
        else:
            self.edge_id = json.loads(args.client_id_list)[0]
        try:
            if args.log_server_url is None or args.log_server_url == "":
                self.log_server_url = "https://open.fedml.ai/fedmlOpsServer/logs/update"
            else:
                self.log_server_url = args.log_server_url
        except Exception as e:
            self.log_server_url = "https://open.fedml.ai/fedmlOpsServer/logs/update"
        self.log_line_index = 0
        self.log_config_file = args.log_file_dir + "/log-config.yaml"
        self.log_config = {}
        self.load_log_config()
        self.origin_log_file_path = (
            self.log_file_dir
            + "/fedavg-cross-silo-run-"
            + str(self.run_id)
            + "-edge-"
            + str(self.edge_id)
            + ".log"
        )
        self.log_file_path = (
            self.log_file_dir
            + "/fedavg-cross-silo-run-"
            + str(self.run_id)
            + "-edge-"
            + str(self.edge_id)
            + "-upload.log"
        )
        if hasattr(self, "should_upload_log_file") and self.should_upload_log_file:
            multiprocessing.Process(target=self.log_thread).start()

    @staticmethod
    def get_instance(args):
        if MLOpsRuntimeLog._log_sdk_instance is None:
            MLOpsRuntimeLog._log_sdk_instance = MLOpsRuntimeLog(args)

        return MLOpsRuntimeLog._log_sdk_instance

    def init_logs(self):
        log_file_path, program_prefix = MLOpsRuntimeLog.build_log_file_path(self.args)
        if hasattr(self, "should_write_log_file") and self.should_write_log_file:
            logging.raiseExceptions = True
            logging.basicConfig(
                filename=log_file_path,
                filemode="w",
                level=logging.INFO,
                format="[" + program_prefix + "] [%(asctime)s] [%(levelname)s] "
                       "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
                datefmt="%a, %d %b %Y %H:%M:%S",
            )
        else:
            logging.raiseExceptions = True
            logging.basicConfig(
                level=logging.INFO,
                format="[" + program_prefix + "] [%(asctime)s] [%(levelname)s] "
                "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
                datefmt="%a, %d %b %Y %H:%M:%S",
            )

    @staticmethod
    def build_log_file_path(args):
        if args.rank == 0:
            edge_id = 0
            program_prefix = "FedML-Server({}) @device-id-{}".format(args.rank, edge_id)
        else:
            edge_id = json.loads(args.client_id_list)[0]
            program_prefix = "FedML-Client({rank}) @device-id-{edge}".format(
                rank=args.rank, edge=edge_id
            )

        os.system("mkdir -p " + args.log_file_dir)
        log_file_path = (
            args.log_file_dir
            + "/fedavg-cross-silo-run-"
            + str(args.run_id)
            + "-edge-"
            + str(edge_id)
            + ".log"
        )

        return log_file_path, program_prefix

    def log_upload(self, run_id, edge_id):
        # read log data from local log file
        log_lines = self.log_read()
        if log_lines is None or len(log_lines) <= 0:
            return

        self.log_line_index += len(log_lines)
        log_upload_request = {
            "run_id": run_id,
            "edge_id": edge_id,
            "logs": log_lines,
            "create_time": time.time(),
            "update_time": time.time(),
            "created_by": str(edge_id),
            "updated_by": str(edge_id),
        }

        log_headers = {'Content-Type': 'application/json', 'Connection': 'close'}

        # send log data to the log server
        _, cert_path = MLOpsConfigs.get_instance(self.args).get_request_params()
        if cert_path is not None:
            cur_source_dir = os.path.dirname(__file__)
            cert_path = os.path.join(cur_source_dir, "ssl", "open.fedml.ai_bundle.crt")
            requests.session().verify = cert_path
            response = requests.post(self.log_server_url, headers=log_headers, json=log_upload_request, verify=True)
        else:
            response = requests.post(self.log_server_url, headers=log_headers, json=log_upload_request)
        if response.status_code != 200:
            # print('Error for sending log data: ' + str(response.status_code))
            self.log_line_index -= len(log_lines)
        else:
            resp_data = response.json()
            # print('The result for sending log data: code %s, content %s' %
            #             (str(response.status_code), str(resp_data)))

    def log_thread(self):
        while True:
            time.sleep(MLOpsRuntimeLog.FED_LOG_UPLOAD_FREQUENCY)
            self.log_upload(self.run_id, self.edge_id)

    def log_relocation(self):
        log_line_count = self.log_line_index
        while log_line_count > 0:
            self.log_file.readline()
            log_line_count -= 1

    def log_open(self):
        try:
            shutil.copyfile(self.origin_log_file_path, self.log_file_path)
            if self.log_file is None:
                self.log_file = open(self.log_file_path, "r")
                self.log_relocation()
        except Exception as e:
            # print("exception at open log file.")
            pass

    def log_read(self):
        self.log_open()

        if self.log_file is None:
            return None

        line_count = 0
        log_lines = []
        while True:
            line_count += 1
            log_line = self.log_file.readline()
            if not log_line:
                break
            log_lines.append(log_line)
        self.log_file.close()
        self.log_file = None
        return log_lines

    @staticmethod
    def __generate_yaml_doc(log_config_object, yaml_file):
        try:
            file = open(yaml_file, "w", encoding="utf-8")
            yaml.dump(log_config_object, file)
            file.close()
        except Exception as e:
            # print("Generate yaml file.")
            pass

    @staticmethod
    def __load_yaml_config(yaml_path):
        """Helper function to load a yaml config file"""
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    def load_log_config(self):
        try:
            self.log_config = self.__load_yaml_config(self.log_config_file)
            self.log_line_index = self.log_config["log_config"]["log_line_index"]
        except Exception as e:
            # print("load_log_config exception")
            pass
