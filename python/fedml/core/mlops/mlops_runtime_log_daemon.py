import argparse
import json

import multiprocess as multiprocessing
import os
import shutil
import threading
import time

import requests
import yaml
from ...core.mlops.mlops_configs import MLOpsConfigs


class MLOpsRuntimeLogProcessor:
    FED_LOG_LINE_NUMS_PER_UPLOADING = 1000
    FED_LOG_UPLOAD_FREQUENCY = 3
    FEDML_LOG_REPORTING_STATUS_FILE_NAME = "log_status"
    FEDML_RUN_LOG_STATUS_DIR = "run_log_status"

    def __init__(self, using_mlops, log_run_id, log_device_id, log_file_dir, log_server_url, in_args=None):
        self.args = in_args
        self.is_log_reporting = False
        self.log_reporting_status_file = os.path.join(log_file_dir,
                                                      MLOpsRuntimeLogProcessor.FEDML_RUN_LOG_STATUS_DIR,
                                                      MLOpsRuntimeLogProcessor.FEDML_LOG_REPORTING_STATUS_FILE_NAME +
                                                      "-" + str(log_run_id) + ".conf")
        os.makedirs(os.path.join(log_file_dir, MLOpsRuntimeLogProcessor.FEDML_RUN_LOG_STATUS_DIR), exist_ok=True)
        self.logger = None
        self.should_upload_log_file = using_mlops
        self.log_file_dir = log_file_dir
        self.log_file = None
        self.run_id = log_run_id
        self.device_id = log_device_id
        self.log_server_url = log_server_url
        self.log_line_index = 0
        self.log_uploaded_line_index = 0
        self.log_config_file = os.path.join(log_file_dir, "log-config.yaml")
        self.log_config = dict()
        self.load_log_config()
        self.origin_log_file_path = os.path.join(self.log_file_dir, "fedml-run-"
                                                 + str(self.run_id)
                                                 + "-edge-"
                                                 + str(self.device_id)
                                                 + ".log")
        self.log_file_path = os.path.join(self.args.log_file_dir, "fedml-run-"
                                          + str(self.run_id)
                                          + "-edge-"
                                          + str(self.device_id)
                                          + "-upload.log")
        self.run_list = list()
        self.log_source = None
        self.log_process_event = None

    def set_log_source(self, source):
        self.log_source = source
        if source is not None:
            self.log_source = str(self.log_source).replace(' ', '')

    @staticmethod
    def build_log_file_path(in_args):
        if in_args.rank == 0:
            if hasattr(in_args, "server_id"):
                log_device_id = in_args.server_id
            else:
                if hasattr(in_args, "edge_id"):
                    log_device_id = in_args.edge_id
                else:
                    log_device_id = 0
            program_prefix = "FedML-Server({}) @device-id-{}".format(in_args.rank, log_device_id)
        else:
            if hasattr(in_args, "client_id"):
                log_device_id = in_args.client_id
            elif hasattr(in_args, "client_id_list"):
                edge_ids = json.loads(in_args.client_id_list)[0]
                if len(edge_ids) > 0:
                    log_device_id = edge_ids[0]
                else:
                    log_device_id = 0
            else:
                if hasattr(in_args, "edge_id"):
                    log_device_id = in_args.edge_id
                else:
                    log_device_id = 0
            program_prefix = "FedML-Client({}) @device-id-{}".format(in_args.rank, log_device_id)

        if not os.path.exists(in_args.log_file_dir):
            os.makedirs(in_args.log_file_dir, exist_ok=True)
        log_file_path = os.path.join(in_args.log_file_dir, "fedml-run-"
                                     + str(in_args.run_id)
                                     + "-edge-"
                                     + str(log_device_id)
                                     + ".log")

        return log_file_path, program_prefix

    def log_upload(self, run_id, device_id):
        # read log data from local log file
        log_lines = self.log_read()
        if log_lines is None or len(log_lines) <= 0:
            return

        line_count = 0
        total_line = len(log_lines)
        send_num_per_req = MLOpsRuntimeLogProcessor.FED_LOG_LINE_NUMS_PER_UPLOADING
        line_start_req = line_count
        while line_count <= total_line:
            line_end_req = line_start_req + send_num_per_req
            if line_end_req >= total_line:
                line_end_req = total_line
            if line_start_req >= line_end_req:
                break

            # Add prefix in exception lines which have not any fedml log prefix
            index = line_start_req
            while index < line_end_req:
                prev_index = index - 1
                if prev_index < 0:
                    prev_index = 0

                if MLOpsRuntimeLogProcessor.should_ignore_log_line(log_lines[index]):
                    log_lines[index] = '\n'
                    index += 1
                    continue

                prev_line_prefix = ''
                prev_line_prefix_list = str(log_lines[prev_index]).split(']')
                if len(prev_line_prefix_list) >= 3:
                    prev_line_prefix = "{}]{}]{}]".format(prev_line_prefix_list[0],
                                                          prev_line_prefix_list[1],
                                                          prev_line_prefix_list[2])

                if not str(log_lines[index]).startswith('[FedML-'):
                    log_line = "{} {}".format(prev_line_prefix, log_lines[index])
                    log_lines[index] = log_line

                index += 1

            # remove the '\n' and '' str
            upload_lines = []
            for line in log_lines[line_start_req:line_end_req]:
                if line != '' and line != '\n':
                    upload_lines.append(line)

            err_list = list()
            for log_index in range(len(upload_lines)):
                log_line = str(upload_lines[log_index])
                if log_line.find(' [ERROR] ') != -1:
                    err_line_dict = {"errLine": self.log_uploaded_line_index + log_index, "errMsg": log_line}
                    err_list.append(err_line_dict)

            log_upload_request = {
                "run_id": run_id,
                "edge_id": device_id,
                "logs": upload_lines,
                "create_time": time.time(),
                "update_time": time.time(),
                "created_by": str(device_id),
                "updated_by": str(device_id)
            }

            if len(err_list) > 0:
                log_upload_request["errors"] = err_list

            if self.log_source is not None and self.log_source != "":
                log_upload_request["source"] = self.log_source

            log_headers = {'Content-Type': 'application/json', 'Connection': 'close'}

            # send log data to the log server
            _, cert_path = MLOpsConfigs.get_instance(self.args).get_request_params()
            if cert_path is not None:
                try:
                    requests.session().verify = cert_path
                    # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}")
                    response = requests.post(
                        self.log_server_url, json=log_upload_request, verify=True, headers=log_headers
                    )
                    # logging.info(f"FedMLDebug POST log to server run_id {run_id}, device_id {device_id}. response.status_code: {response.status_code}")

                except requests.exceptions.SSLError as err:
                    MLOpsConfigs.install_root_ca_file()
                    # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}")
                    response = requests.post(
                        self.log_server_url, json=log_upload_request, verify=True, headers=log_headers
                    )
                    # logging.info(f"FedMLDebug POST log to server run_id {run_id}, device_id {device_id}. response.status_code: {response.status_code}")
            else:
                # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}")
                response = requests.post(self.log_server_url, headers=log_headers, json=log_upload_request)
                # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}. response.status_code: {response.status_code}")
            if response.status_code != 200:
                pass
            else:
                self.log_line_index += (line_end_req - line_start_req)
                self.log_uploaded_line_index += len(upload_lines)
                line_count += (line_end_req - line_start_req)
                line_start_req = line_end_req
                self.save_log_config()
                resp_data = response.json()

    @staticmethod
    def should_ignore_log_line(log_line):
        #  if str is empty, then continue, will move it later
        if str(log_line) == '' or str(log_line) == '\n':
            return True

        #  if the str has prefix but contains nothing,
        #  then signed it as '\n', will move it later
        cur_line_list = str(log_line).split(']')
        if str(log_line).startswith('[FedML-') and \
                len(cur_line_list) == 5 and (cur_line_list[4] == ' \n'):
            return True

        return False

    def log_process(self, process_event):
        self.log_process_event = process_event

        while not self.should_stop():
            try:
                time.sleep(MLOpsRuntimeLogProcessor.FED_LOG_UPLOAD_FREQUENCY)
                self.log_upload(self.run_id, self.device_id)
            except Exception as e:
                pass

        self.log_upload(self.run_id, self.device_id)
        print("FedDebug log_process STOPPED")

    def log_relocation(self):
        log_line_count = self.log_line_index
        self.log_uploaded_line_index = self.log_line_index
        while log_line_count > 0:
            line = self.log_file.readline()
            if line is None:
                break
            if MLOpsRuntimeLogProcessor.should_ignore_log_line(line):
                self.log_uploaded_line_index -= 1
            log_line_count -= 1

        if log_line_count != 0:
            self.log_line_index -= log_line_count
            if self.log_line_index < 0:
                self.log_line_index = 0

    def log_open(self):
        try:
            shutil.copyfile(self.origin_log_file_path, self.log_file_path)
            if self.log_file is None:
                self.log_file = open(self.log_file_path, "r")
                self.log_relocation()
        except Exception as e:
            pass

    def log_read(self):
        self.log_open()

        if self.log_file is None:
            return None

        line_count = 0
        log_lines = []
        while True:
            log_line = self.log_file.readlines()
            if len(log_line) <= 0:
                break
            line_count += len(log_line)
            log_lines.extend(log_line)
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
            pass

    @staticmethod
    def __load_yaml_config(yaml_path):
        """Helper function to load a yaml config file"""
        with open(yaml_path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError("Yaml error - check yaml file")

    def save_log_config(self):
        try:
            log_config_key = "log_config_{}_{}".format(self.run_id, self.device_id)
            self.log_config[log_config_key] = dict()
            self.log_config[log_config_key]["log_line_index"] = self.log_line_index
            MLOpsRuntimeLogProcessor.__generate_yaml_doc(self.log_config, self.log_config_file)
        except Exception as e:
            pass

    def load_log_config(self):
        try:
            log_config_key = "log_config_{}_{}".format(self.run_id, self.device_id)
            self.log_config = self.__load_yaml_config(self.log_config_file)
            self.log_line_index = self.log_config[log_config_key]["log_line_index"]
        except Exception as e:
            pass

    def should_stop(self):
        if self.log_process_event is not None and self.log_process_event.is_set():
            return True

        return False


class MLOpsRuntimeLogDaemon:
    _log_sdk_instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(MLOpsRuntimeLogDaemon, "_instance"):
            with MLOpsRuntimeLogDaemon._instance_lock:
                if not hasattr(MLOpsRuntimeLogDaemon, "_instance"):
                    MLOpsRuntimeLogDaemon._instance = object.__new__(cls)
        return MLOpsRuntimeLogDaemon._instance

    def __init__(self, in_args):
        self.args = in_args

        if in_args.role == "server":
            if hasattr(in_args, "server_id"):
                self.edge_id = in_args.server_id
            else:
                if hasattr(in_args, "edge_id"):
                    self.edge_id = in_args.edge_id
                else:
                    self.edge_id = 0
        else:
            if hasattr(in_args, "client_id"):
                self.edge_id = in_args.client_id
            elif hasattr(in_args, "client_id_list"):
                edge_ids = json.loads(in_args.client_id_list)
                if len(edge_ids) > 0:
                    self.edge_id = edge_ids[0]
                else:
                    self.edge_id = 0
            else:
                if hasattr(in_args, "edge_id"):
                    self.edge_id = in_args.edge_id
                else:
                    self.edge_id = 0

        try:
            if self.args.log_server_url is None or self.args.log_server_url == "":
                self.log_server_url = "https://open.fedml.ai/fedmlLogsServer/logs/update"
            else:
                self.log_server_url = self.args.log_server_url
        except Exception as e:
            self.log_server_url = "https://open.fedml.ai/fedmlLogsServer/logs/update"

        self.log_file_dir = self.args.log_file_dir
        self.log_child_process_list = list()
        self.log_child_process = None
        self.log_process_event = None

    @staticmethod
    def get_instance(args):
        if MLOpsRuntimeLogDaemon._log_sdk_instance is None:
            MLOpsRuntimeLogDaemon._log_sdk_instance = MLOpsRuntimeLogDaemon(args)
            MLOpsRuntimeLogDaemon._log_sdk_instance.log_source = None

        return MLOpsRuntimeLogDaemon._log_sdk_instance

    def set_log_source(self, source):
        self.log_source = source

    def start_log_processor(self, log_run_id, log_device_id):
        log_processor = MLOpsRuntimeLogProcessor(self.args.using_mlops, log_run_id,
                                                 log_device_id, self.log_file_dir,
                                                 self.log_server_url,
                                                 in_args=self.args)
        log_processor.set_log_source(self.log_source)
        if self.log_process_event is None:
            self.log_process_event = multiprocessing.Event()
        self.log_process_event.clear()
        log_processor.log_process_event = self.log_process_event
        self.log_child_process = multiprocessing.Process(target=log_processor.log_process,
                                                         args=(self.log_process_event,))
        # process = threading.Thread(target=log_processor.log_process)
        if self.log_child_process is not None:
            self.log_child_process.start()
            try:
                self.log_child_process_list.index((self.log_child_process, log_run_id, log_device_id))
            except ValueError as ex:
                self.log_child_process_list.append((self.log_child_process, log_run_id, log_device_id))

    def stop_log_processor(self, log_run_id, log_device_id):
        if log_run_id is None or log_device_id is None:
            return

        # logging.info(f"FedMLDebug. stop log processor. log_run_id = {log_run_id}, log_device_id = {log_device_id}")
        for (log_child_process, run_id, device_id) in self.log_child_process_list:
            if str(run_id) == str(log_run_id) and str(device_id) == str(log_device_id):
                if self.log_process_event is not None:
                    self.log_process_event.set()
                else:
                    log_child_process.terminate()
                break

    def stop_all_log_processor(self):
        for (log_child_process, _, _) in self.log_child_process_list:
            if self.log_process_event is not None:
                self.log_process_event.set()
            else:
                log_child_process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_file_dir", "-log", help="log file dir")
    parser.add_argument("--rank", "-r", type=str, default="1")
    parser.add_argument("--client_id_list", "-cil", type=str, default="[]")
    parser.add_argument("--log_server_url", "-lsu", type=str, default="http://")

    args = parser.parse_args()
    setattr(args, "using_mlops", True)
    setattr(args, "config_version", "local")

    run_id = 9998
    device_id = 1
    MLOpsRuntimeLogDaemon.get_instance(args).start_log_processor(run_id, device_id)

    while True:
        time.sleep(1)
        # MLOpsRuntimeLogDaemon.get_instance(args).stop_log_processor(run_id, device_id)
