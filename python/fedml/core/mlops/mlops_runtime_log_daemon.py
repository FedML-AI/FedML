import argparse
import logging
import os
import shutil
import threading
import time

import multiprocess as multiprocessing
import requests
import yaml

import fedml
from fedml.computing.scheduler.comm_utils.run_process_utils import RunProcessUtils
from fedml.core.mlops.mlops_utils import MLOpsLoggingUtils
from ...core.mlops.mlops_configs import MLOpsConfigs


class MLOpsRuntimeLogProcessor:
    FED_LOG_LINE_NUMS_PER_UPLOADING = 1000
    FED_LOG_UPLOAD_FREQUENCY = 3
    FED_LOG_UPLOAD_S3_FREQUENCY = 30
    FEDML_LOG_REPORTING_STATUS_FILE_NAME = "log_status"
    FEDML_RUN_LOG_STATUS_DIR = "run_log_status"

    ENABLE_UPLOAD_LOG_USING_MQTT = False

    def __init__(
            self, using_mlops, log_run_id, log_device_id, log_file_dir, log_server_url, in_args=None,
            log_file_prefix=None
    ):
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
        self.run_id = log_run_id
        self.device_id = log_device_id
        self.log_server_url = log_server_url
        self.file_rotate_count = 0
        self.log_config_file = os.path.join(log_file_dir, MLOpsLoggingUtils.LOG_CONFIG_FILE)
        self.origin_log_file_path = os.path.join(self.log_file_dir, "fedml-run-"
                                                 + ("" if log_file_prefix is None else f"{log_file_prefix}-")
                                                 + str(self.run_id)
                                                 + "-edge-"
                                                 + str(self.device_id)
                                                 + ".log")
        # self.log_files_to_upload.add(self.origin_log_file_path)
        self.log_file_path = os.path.join(self.args.log_file_dir, "fedml-run-"
                                          + ("" if log_file_prefix is None else f"{log_file_prefix}-")
                                          + str(self.run_id)
                                          + "-edge-"
                                          + str(self.device_id)
                                          + "-upload.log")
        self.log_source = None
        self.log_process_event = None

        self.should_split_by_replica = False

        if log_file_prefix is not None and log_file_prefix == "endpoint":
            self.should_split_by_replica = True

    def set_log_source(self, source):
        self.log_source = source
        if source is not None:
            self.log_source = str(self.log_source).replace(' ', '')

    def log_upload(self, run_id, device_id):
        # Fetch Log Lines
        file_index, log_lines = self.fetch_logs()
        uploaded_file_index = file_index
        if log_lines is None or len(log_lines) <= 0:
            return

        line_count = 0
        total_line = len(log_lines)
        send_num_per_req = MLOpsRuntimeLogProcessor.FED_LOG_LINE_NUMS_PER_UPLOADING
        line_start_req = line_count
        while line_count <= total_line:
            line_end_req = line_start_req + send_num_per_req
            line_end_req = total_line if line_end_req > total_line else line_end_req
            if line_start_req >= line_end_req:
                break

            self.__format_log_lines(log_lines, line_start_req, line_end_req)
            upload_lines, err_list = self.__preprocess_logs(log_lines, line_start_req, line_end_req)

            if not self.should_split_by_replica:
                log_upload_request = self.__prepare_request(upload_lines, err_list, run_id, device_id)
                if MLOpsRuntimeLogProcessor.ENABLE_UPLOAD_LOG_USING_MQTT:
                    fedml.core.mlops.log_run_logs(log_upload_request, run_id=run_id)
                else:
                    upload_successful = self.__upload(log_upload_request)
                    if not upload_successful:
                        return
            else:
                # Report log data of each replica separately
                replica_id_to_lines = self.__split_diff_replica_lines(upload_lines)
                err_replica_id_to_lines = self.__split_diff_replica_lines(err_list)

                for replica_id, lines in replica_id_to_lines.items():
                    err_list = []
                    if replica_id in err_replica_id_to_lines:
                        err_list = err_replica_id_to_lines[replica_id]
                    log_upload_request = self.__prepare_request(lines, err_list, run_id, device_id, replica_id)
                    if MLOpsRuntimeLogProcessor.ENABLE_UPLOAD_LOG_USING_MQTT:
                        fedml.core.mlops.log_run_logs(log_upload_request, run_id=run_id)
                    else:
                        upload_successful = self.__upload(log_upload_request)
                        if not upload_successful:
                            return

            num_lines_uploaded = line_end_req - line_start_req
            uploaded_file_index += num_lines_uploaded
            line_count += num_lines_uploaded
            line_start_req = line_end_req

            # Update the uploaded file index
            MLOpsLoggingUtils.acquire_lock()
            config_data = MLOpsLoggingUtils.load_log_config(run_id, device_id,
                                                            self.log_config_file)

            config_data[self.file_rotate_count].uploaded_file_index = uploaded_file_index
            MLOpsLoggingUtils.save_log_config(run_id=run_id, device_id=device_id,
                                              log_config_file=self.log_config_file,
                                              config_data=config_data)
            MLOpsLoggingUtils.release_lock()

    @staticmethod
    def __format_log_lines(log_lines: list, line_start_req: int, line_end_req: int):
        # Format Log Lines
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

    @staticmethod
    def __preprocess_logs(log_lines: list, line_start_req: int, line_end_req: int) -> (list, list):
        # remove the '\n' and '' str
        upload_lines, err_lines = list(), list()
        for line in log_lines[line_start_req:line_end_req]:
            if line != '' and line != '\n':
                upload_lines.append(line)
                log_line = str(line)
                if log_line.find(' [ERROR] ') != -1:
                    err_line_dict = {"errLine": line_start_req, "errMsg": log_line}
                    err_lines.append(err_line_dict)
        return upload_lines, err_lines

    @staticmethod
    def __split_diff_replica_lines(upload_lines):
        # Return replica_id_to_lines = {replica_id: split_lines}
        # Add replica_id to log_upload_request (i.e. upload log data of each replica separately)
        # Parse the first square brackets. e.g. [FedML-Client(0) @device-id-0 @replica-rank-0]
        replica_rank = 0
        replica_id_to_lines = dict()
        for line in upload_lines:
            # Try to find replica-rank in the line, default is 0
            if line.find("[") != -1 and line.find("]") != -1:
                content_inside_brackets = line[line.find("[") + 1:line.find("]")]
                if content_inside_brackets.find("replica-rank") != -1:
                    replica_rank = int(content_inside_brackets.split("-")[-1])

            # id starts from 1
            replica_id = replica_rank + 1

            # Split to different keys
            if replica_id not in replica_id_to_lines:
                replica_id_to_lines[replica_id] = []
            replica_id_to_lines[replica_id].append(line)
        return replica_id_to_lines

    def __prepare_request(self, upload_lines, err_list, run_id, device_id, replica_id=None) -> dict:
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

        if replica_id is not None:
            log_upload_request["replica_id"] = f"{device_id}_{replica_id}"

        return log_upload_request

    def __upload(self, log_upload_request) -> bool:
        log_headers = {'Content-Type': 'application/json', 'Connection': 'close'}

        # send log data to the log server
        _, cert_path = MLOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    self.log_server_url, json=log_upload_request, verify=True, headers=log_headers
                )

            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    self.log_server_url, json=log_upload_request, verify=True, headers=log_headers
                )
        else:
            response = requests.post(self.log_server_url, headers=log_headers, json=log_upload_request)
        if response.status_code != 200:
            logging.error(f"Failed to upload log to server. run_id {self.run_id}, device_id {self.device_id}. "
                          f"response.status_code: {response.status_code}")
            return False
        return True

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
        logging.info(f"Log uploading process id {os.getpid()}, run id {self.run_id}, edge id {self.device_id}")
        self.log_process_event = process_event

        only_push_artifact = False
        log_artifact_time_counter = 0
        log_file_prev_size = 0
        artifact_url_logged = False
        while not self.should_stop():
            try:
                time.sleep(MLOpsRuntimeLogProcessor.FED_LOG_UPLOAD_FREQUENCY)

                self.log_upload(self.run_id, self.device_id)

                log_artifact_time_counter += MLOpsRuntimeLogProcessor.FED_LOG_UPLOAD_FREQUENCY
                if log_artifact_time_counter >= MLOpsRuntimeLogProcessor.FED_LOG_UPLOAD_S3_FREQUENCY:
                    log_artifact_time_counter = 0
                    log_file_current_size = os.path.getsize(self.log_file_path) if os.path.exists(
                        self.log_file_path) else 0
                    if log_file_prev_size != log_file_current_size:
                        upload_result, artifact_storage_url = self.upload_log_file_as_artifact(
                            only_push_artifact=only_push_artifact)
                        if upload_result:
                            only_push_artifact = True
                            if artifact_url_logged is False:
                                artifact_url_logged = True
                                fedml.mlops.log_run_log_lines(
                                    self.run_id, self.device_id, [f"The original log url is {artifact_storage_url}"],
                                    log_source=self.log_source
                                )
                        log_file_prev_size = os.path.getsize(self.log_file_path) if os.path.exists(
                            self.log_file_path) else 0
            except Exception as e:
                log_artifact_time_counter = 0
                pass

        self.log_upload(self.run_id, self.device_id)
        self.upload_log_file_as_artifact(only_push_artifact=True)
        print("Log Process exits normally.")

    def fetch_file_path_and_index(self) -> (str, int):
        try:
            upload_file_index = None
            MLOpsLoggingUtils.acquire_lock()
            config_data = MLOpsLoggingUtils.load_log_config(run_id=self.run_id, device_id=self.device_id,
                                                            log_config_file=self.log_config_file)
            MLOpsLoggingUtils.release_lock()
            if config_data is not None:
                config_len = len(config_data)
                upload_file_config = config_data.get(self.file_rotate_count, None)
                if upload_file_config is not None:
                    file_path, uploaded_file_index = upload_file_config.file_path, upload_file_config.uploaded_file_index
                    shutil.copyfile(file_path, self.log_file_path)
                    if MLOpsRuntimeLogProcessor.is_file_rotated(self.log_file_path, uploaded_file_index, config_len,
                                                                self.file_rotate_count):
                        MLOpsLoggingUtils.acquire_lock()
                        config_data = MLOpsLoggingUtils.load_log_config(run_id=self.run_id, device_id=self.device_id,
                                                                        log_config_file=self.log_config_file)
                        config_data[self.file_rotate_count].upload_complete = True
                        MLOpsLoggingUtils.save_log_config(run_id=self.run_id, device_id=self.device_id,
                                                          log_config_file=self.log_config_file, config_data=config_data)
                        MLOpsLoggingUtils.release_lock()
                        self.file_rotate_count += 1
                        # Re-fetch file path and index if file is rotated
                        return self.fetch_file_path_and_index()
                    return uploaded_file_index

            return upload_file_index
        except Exception as e:
            raise ValueError(f"Failed to open log file. Exception: {e}")
        finally:
            MLOpsLoggingUtils.release_lock()

    @staticmethod
    def is_file_rotated(file_path, uploaded_file_index, config_len, rotate_count):
        # move the log file pointer to the last uploaded line
        if config_len == rotate_count + 1:
            return False

        with open(file_path, "r") as f:
            lines = f.readlines()
            if len(lines) > uploaded_file_index:
                return False
            return True

    def fetch_logs(self) -> (str, int, list):
        log_lines = []
        file_index = self.fetch_file_path_and_index()
        if file_index is not None:
            with open(self.log_file_path, "r") as f:
                lines = f.readlines()
                log_lines.extend(lines[file_index:])
        return file_index, log_lines

    @staticmethod
    def __generate_yaml_doc(log_config_object, yaml_file):
        try:
            file = open(yaml_file, "w", encoding="utf-8")
            yaml.dump(log_config_object, file)
            file.close()
        except Exception as e:
            pass

    def should_stop(self):
        if self.log_process_event is not None and self.log_process_event.is_set():
            return True

        return False

    def upload_log_file_as_artifact(self, only_push_artifact=False):
        try:
            if not os.path.exists(self.log_file_path):
                return False, ""

            log_file_name = "{}".format(os.path.basename(self.log_file_path))
            log_file_name_no_ext = os.path.splitext(os.path.basename(self.log_file_path))[0]
            artifact = fedml.mlops.Artifact(name=log_file_name_no_ext, type=fedml.mlops.ARTIFACT_TYPE_NAME_LOG)
            artifact.add_file(self.log_file_path)

            artifact_storage_url = fedml.core.mlops.log_mlops_running_logs(
                artifact, run_id=self.run_id, edge_id=self.device_id, only_push_artifact=only_push_artifact)

            return True, artifact_storage_url
        except Exception as e:
            return False, ""


class MLOpsRuntimeLogDaemon:
    _log_sdk_instance = None
    _instance_lock = threading.Lock()

    ENABLE_SYNC_LOG_TO_MASTER = False

    def __new__(cls, *args, **kwargs):
        if not hasattr(MLOpsRuntimeLogDaemon, "_instance"):
            with MLOpsRuntimeLogDaemon._instance_lock:
                if not hasattr(MLOpsRuntimeLogDaemon, "_instance"):
                    MLOpsRuntimeLogDaemon._instance = object.__new__(cls)
        return MLOpsRuntimeLogDaemon._instance

    def __init__(self, in_args):
        self.args = in_args
        self.edge_id = MLOpsLoggingUtils.get_edge_id_from_args(self.args)
        url = fedml._get_backend_service()
        try:
            if self.args.log_server_url is None or self.args.log_server_url == "":
                self.log_server_url = f"{url}/fedmlLogsServer/logs/update"
            else:
                self.log_server_url = self.args.log_server_url
        except Exception as e:
            self.log_server_url = f"{url}/fedmlLogsServer/logs/update"

        if MLOpsRuntimeLogDaemon.ENABLE_SYNC_LOG_TO_MASTER:
            self.log_server_url = f"http://localhost:42000/fedml/api/v2/fedmlLogsServer/logs/update"

        self.log_file_dir = self.args.log_file_dir
        self.log_child_process_list = list()
        self.log_process_event_map = dict()

    def set_log_source(self, source):
        self.log_source = source

    def start_log_processor(self, log_run_id, log_device_id, log_source=None, log_file_prefix=None):
        log_processor = MLOpsRuntimeLogProcessor(self.args.using_mlops, log_run_id,
                                                 log_device_id, self.log_file_dir,
                                                 self.log_server_url,
                                                 in_args=self.args, log_file_prefix=log_file_prefix)
        if log_source is not None:
            log_processor.set_log_source(log_source)
        else:
            log_processor.set_log_source(self.log_source)
        event_map_id = self.get_event_map_id(log_run_id, log_device_id)
        if self.log_process_event_map.get(event_map_id, None) is None:
            self.log_process_event_map[event_map_id] = multiprocessing.Event()
        self.log_process_event_map[event_map_id].clear()
        log_processor.log_process_event = self.log_process_event_map[event_map_id]
        log_child_process = fedml.get_multiprocessing_context().Process(
            target=log_processor.log_process, args=(self.log_process_event_map[event_map_id],))
        # process = threading.Thread(target=log_processor.log_process)
        # process.start()
        if log_child_process is not None:
            log_child_process.start()
            try:
                self.log_child_process_list.index((log_child_process, log_run_id, log_device_id))
            except ValueError as ex:
                self.log_child_process_list.append((log_child_process, log_run_id, log_device_id))

    def stop_log_processor(self, log_run_id, log_device_id):
        if log_run_id is None or log_device_id is None:
            return

        # logging.info(f"FedMLDebug. stop log processor. log_run_id = {log_run_id}, log_device_id = {log_device_id}")
        event_map_id = self.get_event_map_id(log_run_id, log_device_id)
        for (log_child_process, run_id, device_id) in self.log_child_process_list:
            if str(run_id) == str(log_run_id) and str(device_id) == str(log_device_id):
                if self.log_process_event_map.get(event_map_id, None) is not None:
                    self.log_process_event_map[event_map_id].set()
                else:
                    log_child_process.terminate()
                self.log_child_process_list.remove((log_child_process, run_id, device_id))
                break

    def stop_all_log_processor(self):
        for (log_child_process, run_id, device_id) in self.log_child_process_list:
            event_map_id = self.get_event_map_id(run_id, device_id)
            if self.log_process_event_map.get(event_map_id, None) is not None:
                self.log_process_event_map[event_map_id].set()
            else:
                log_child_process.terminate()

    def is_log_processor_running(self, in_run_id, in_device_id):
        for (log_child_process, log_run_id, log_device_id) in self.log_child_process_list:
            if str(in_run_id) == str(log_run_id) and str(in_device_id) == str(log_device_id) and \
                    log_child_process is not None and RunProcessUtils.is_process_running(log_child_process.pid):
                return True

        return False

    @staticmethod
    def get_instance(in_args):
        if MLOpsRuntimeLogDaemon._log_sdk_instance is None:
            MLOpsRuntimeLogDaemon._log_sdk_instance = MLOpsRuntimeLogDaemon(in_args)
            MLOpsRuntimeLogDaemon._log_sdk_instance.log_source = None

        return MLOpsRuntimeLogDaemon._log_sdk_instance

    @staticmethod
    def get_event_map_id(log_run_id, log_device_id):
        return f"{log_run_id}_{log_device_id}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_file_dir", "-log", help="log file dir")
    parser.add_argument("--rank", "-r", type=str, default="1")
    parser.add_argument("--client_id_list", "-cil", type=str, default="[]")
    parser.add_argument("--log_server_url", "-lsu", type=str, default="http://")

    args = parser.parse_args()
    setattr(args, "using_mlops", True)
    setattr(args, "config_version", "local")
    setattr(args, "log_file_dir", "/Volumes/Projects/FedML/python/fedml/core/mlops/test/logs")
    setattr(args, "run_id", "10")
    setattr(args, "edge_id", "11")
    setattr(args, "role", "client")
    setattr(args, "config_version", "local")
    setattr(args, "using_mlops", True)
    setattr(args, "log_server_url", "http://localhost:8080")


    run_id = 9998
    device_id = 1
    MLOpsRuntimeLogDaemon.get_instance(args).start_log_processor(run_id, device_id)

    while True:
        time.sleep(1)
        # MLOpsRuntimeLogDaemon.get_instance(args).stop_log_processor(run_id, device_id)
