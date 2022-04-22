import logging
import multiprocessing
import os
import sys
import json
import threading
import time

import requests
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))


class FedLogsSDK:
    FED_LOG_LINE_NUMS_PER_UPLOADING = 100

    _log_sdk_instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(FedLogsSDK, "_instance"):
            with FedLogsSDK._instance_lock:
                if not hasattr(FedLogsSDK, "_instance"):
                    FedLogsSDK._instance = object.__new__(cls)
        return FedLogsSDK._instance

    def __init__(self, args):
        self.args = args
        self.should_write_log_file = True
        self.should_upload_log_file = True
        self.log_file_dir = args.log_file_dir
        self.log_file = None
        self.run_id = args.run_id
        if args.silo_rank == 0:
            self.edge_id = 0
        else:
            self.edge_id = json.loads(args.client_ids)[0]
        if args.log_server_url is None or args.log_server_url == "":
            self.log_server_url ="https://open.fedml.ai/fedmlOpsServer/logs/update"
        else:
            self.log_server_url = args.log_server_url
        self.log_line_index = 0
        self.log_config_file = args.log_file_dir + "/log-config.yaml"
        self.log_config = {}
        self.load_log_config()
        self.origin_log_file_path = self.log_file_dir + "/fedavg-cross-silo-run-" + str(self.run_id) + \
                                    "-edge-" + str(self.edge_id) + ".log"
        self.log_file_path = self.log_file_dir + "/fedavg-cross-silo-run-" + str(self.run_id) + \
                             "-edge-" + str(self.edge_id) + "-upload.log"
        if self.should_upload_log_file:
            multiprocessing.Process(target=self.log_thread).start()

    @staticmethod
    def get_instance(args):
        if FedLogsSDK._log_sdk_instance is None:
            FedLogsSDK._log_sdk_instance = FedLogsSDK(args)

        return FedLogsSDK._log_sdk_instance

    def init_logs(self):
        log_file_path, program_prefix = FedLogsSDK.build_log_file_path(self.args)
        if self.should_write_log_file:
            logging.basicConfig(
                filename=log_file_path,
                filemode="w",
                level=logging.INFO,
                format=program_prefix + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                datefmt="%a, %d %b %Y %H:%M:%S",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=program_prefix + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                datefmt="%a, %d %b %Y %H:%M:%S",
            )

    @staticmethod
    def build_log_file_path(args):
        if args.silo_rank == 0:
            edge_id = 0
            program_prefix = "FedML-Server({}) @device-id-{}".format(args.silo_rank, edge_id)
        else:
            edge_id = json.loads(args.client_ids)[0]
            program_prefix = "FedML-Client({rank}) @device-id-{edge}".format(rank=args.silo_rank, edge=edge_id)

        os.system("mkdir -p " + args.log_file_dir)
        client_ids = json.loads(args.client_ids)
        log_file_path = args.log_file_dir + "/fedavg-cross-silo-run-" + str(args.run_id) + \
                        "-edge-" + str(edge_id) + ".log"

        return log_file_path, program_prefix

    def log_upload(self, run_id, edge_id):
        # read log data from local log file
        log_lines = self.log_read()
        if log_lines is None or len(log_lines) <= 0:
            return

        self.log_line_index += len(log_lines)
        log_upload_request = {"run_id": run_id, "edge_id": edge_id, "logs": log_lines,
                              "create_time": time.time(), "update_time": time.time(),
                              "created_by": str(edge_id), "updated_by": str(edge_id)}

        # set request header with the application/json format
        log_headers = {'Content-Type': 'application/json'}

        # send log data to the log server
        response = requests.post(self.log_server_url, headers=log_headers, json=log_upload_request, verify=False)
        if response.status_code != 200:
            # print('Error for sending log data: ' + str(response.status_code))
            self.log_line_index -= len(log_lines)
        else:
            resp_data = response.json()
            # print('The result for sending log data: code %s, content %s' %
            #             (str(response.status_code), str(resp_data)))

    def log_thread(self):
        while True:
            time.sleep(10)
            self.log_upload(self.run_id, self.edge_id)

    def log_relocation(self):
        log_line_count = self.log_line_index
        while log_line_count > 0:
            self.log_file.readline()
            log_line_count -= 1

    def log_open(self):
        try:
            os.system("cp -f " + self.origin_log_file_path + " " + self.log_file_path)
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
            file = open(yaml_file, 'w', encoding='utf-8')
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
