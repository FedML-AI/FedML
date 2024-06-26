import json
import logging
import multiprocessing
import os
import platform
import random
import shutil
import time
import traceback
import zipfile
import queue

import fedml
from ..comm_utils.constants import SchedulerConstants
from ..comm_utils.job_utils import JobRunnerUtils, DockerArgs
from ..scheduler_entry.constants import Constants
from ....core.mlops import MLOpsMetrics, MLOpsRuntimeLogDaemon
from ....core.mlops.mlops_device_perfs import MLOpsDevicePerfStats
from ..comm_utils.yaml_utils import load_yaml_config
from .general_constants import GeneralConstants
from ..comm_utils.sys_utils import get_python_program
from ..comm_utils import sys_utils
from ....core.mlops.mlops_utils import MLOpsUtils
from ..scheduler_core.message_center import FedMLMessageCenter
from ..scheduler_core.status_center import FedMLStatusCenter
from abc import ABC, abstractmethod
import ssl


class RunnerError(Exception):
    """ Runner stopped. """
    pass


class RunnerCompletedError(Exception):
    """ Runner completed. """
    pass


class FedMLSchedulerBaseJobRunner(ABC):

    def __init__(self, args, edge_id=0, request_json=None, agent_config=None, run_id=0,
                 cuda_visible_gpu_ids_str=None, is_master_runner=False,
                 agent_data_dir=None, agent_package_download_dir=None,
                 agent_package_unzip_dir=None, agent_log_file_dir=None):
        self.args = args
        self.is_master_runner = is_master_runner
        self.agent_data_dir = agent_data_dir
        self.agent_package_download_dir = agent_package_download_dir
        self.agent_package_unzip_dir = agent_package_unzip_dir
        self.agent_log_file_dir = agent_log_file_dir
        self.prev_download_progress = 0
        self.run_process_event = None
        self.run_process_completed_event = None
        self.run_process = None
        self.running_request_json = dict()
        self.start_request_json = None
        self.edge_id = edge_id
        self.edge_user_name = None
        self.edge_extra_url = None
        self.run_id = run_id
        self.unique_device_id = args.unique_device_id
        self.request_json = request_json
        self.version = args.version
        self.device_id = args.device_id
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]
        self.agent_config = agent_config
        self.mlops_metrics = None
        self.status_reporter = None
        self.ntp_offset = MLOpsUtils.get_ntp_offset()
        self.server_id = None
        self.fedml_config_object = None
        self.package_type = SchedulerConstants.JOB_PACKAGE_TYPE_DEFAULT
        self.cuda_visible_gpu_ids_str = cuda_visible_gpu_ids_str
        self.user_name = None
        self.general_edge_id = None
        self.message_center = None
        self.status_center = None
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES = {
            "${FEDSYS.RUN_ID}": "",
            "${FEDSYS.PRIVATE_LOCAL_DATA}": "",
            "${FEDSYS.CLIENT_ID_LIST}": "",
            "${FEDSYS.SYNTHETIC_DATA_URL}": "",
            "${FEDSYS.IS_USING_LOCAL_DATA}": "",
            "${FEDSYS.CLIENT_NUM}": "",
            "${FEDSYS.CLIENT_INDEX}": "",
            "${FEDSYS.CLIENT_OBJECT_LIST}": "",
            "${FEDSYS.LOG_SERVER_URL}": "",
        }
        self.is_deployment_runner = False

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def build_dynamic_constrain_variables(self, run_id, run_config):
        data_config = run_config.get("data_config", {})
        server_edge_id_list = self.request_json["edgeids"]
        local_edge_id_list = list()
        local_edge_id_list.append(int(self.edge_id))
        is_using_local_data = 0
        private_data_dir = data_config.get("privateLocalData", "")
        synthetic_data_url = data_config.get("syntheticDataUrl", "")
        edges = self.request_json["edges"]
        # if private_data_dir is not None \
        #         and len(str(private_data_dir).strip(' ')) > 0:
        #     is_using_local_data = 1
        if private_data_dir is None or len(str(private_data_dir).strip(" ")) <= 0:
            params_config = run_config.get("parameters", None)
            private_data_dir = self.agent_data_dir
        if synthetic_data_url is None or len(str(synthetic_data_url)) <= 0:
            synthetic_data_url = private_data_dir

        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.RUN_ID}"] = run_id
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.PRIVATE_LOCAL_DATA}"] = private_data_dir.replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_ID_LIST}"] = \
            str(self.get_client_id_list(server_edge_id_list)).replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.SYNTHETIC_DATA_URL}"] = synthetic_data_url.replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.IS_USING_LOCAL_DATA}"] = str(is_using_local_data)
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_NUM}"] = len(server_edge_id_list)
        if not self.is_master_runner:
            self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_INDEX}"] = 1
            for cur_index, id_value in enumerate(server_edge_id_list):
                if str(id_value) == str(self.edge_id):
                    self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_INDEX}"] = cur_index + 1
                    break
        client_objects = str(json.dumps(edges))
        client_objects = client_objects.replace(" ", "").replace("\n", "").replace('"', '\\"')
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_OBJECT_LIST}"] = client_objects
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.LOG_SERVER_URL}"] = self.agent_config["ml_ops_config"][
            "LOG_SERVER_URL"
        ]

    def get_client_id_list(self, server_edge_id_list):
        local_edge_id_list = list()
        local_edge_id_list.append(int(self.edge_id))
        return local_edge_id_list

    @staticmethod
    def unzip_file(zip_file, unzip_file_path) -> str:
        unzipped_file_name = ""
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                # Make sure the unzipped file is a directory.
                if zipf.namelist()[0].endswith("/"):
                    unzipped_file_name = zipf.namelist()[0]
        else:
            raise Exception("Invalid zip file {}".format(zip_file))

        return unzipped_file_name

    def package_download_progress(self, count, blksize, filesize):
        self.check_runner_stop_event()

        downloaded = count * blksize
        downloaded = filesize if downloaded > filesize else downloaded
        progress = (downloaded / filesize * 100) if filesize != 0 else 0
        progress_int = int(progress)
        downloaded_kb = format(downloaded / 1024, '.2f')

        # Since this hook function is stateless, we need a state to avoid print progress repeatedly.
        if count == 0:
            self.prev_download_progress = 0
        if progress_int != self.prev_download_progress and progress_int % 5 == 0:
            self.prev_download_progress = progress_int
            logging.info("package downloaded size {} KB, progress {}%".format(downloaded_kb, progress_int))

    def download_package_proc(self, package_url, local_package_file, completed_event, info_queue):
        import requests
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'}
        user_agent_list = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        ]
        for _ in user_agent_list:
            user_agent = random.choice(user_agent_list)
            headers = {'User-Agent': user_agent}

        # Set the stream to true so that we can reduce the memory footprint when downloading large files.
        request = requests.get(package_url, headers=headers, timeout=(10, 15), stream=True)
        with open(local_package_file, 'wb') as f:
            # 1024 * 1024 is 1MiB
            download_size = 1024 * 1024
            total_size = 0
            for chunk in request.iter_content(download_size):
                # Write the chunk to the file
                written_size = f.write(chunk)
                total_size += written_size
                logging.info("package downloaded size %.2f KB", total_size/1024)
                info_queue.put(time.time())
        completed_event.set()

    def retrieve_and_unzip_package(self, package_name, package_url):
        local_package_path = self.agent_package_download_dir
        os.makedirs(local_package_path, exist_ok=True)
        filename, filename_without_extension, file_extension = GeneralConstants.get_filename_and_extension(package_url)
        local_package_file = os.path.join(
            local_package_path, f"fedml_run_{self.run_id}_{self.edge_id}_{filename_without_extension}")
        if os.path.exists(local_package_file):
            os.remove(local_package_file)
        ssl._create_default_https_context = ssl._create_unverified_context

        # Open a process to download the package so that we can avoid the request is blocked and check the timeout.
        from multiprocessing import Process
        completed_event = multiprocessing.Event()
        info_queue = multiprocessing.Manager().Queue()
        if platform.system() == "Windows":
            download_process = multiprocessing.Process(
                target=self.download_package_proc,
                args=(package_url, local_package_file, completed_event, info_queue))
        else:
            download_process = fedml.get_process(
                target=self.download_package_proc,
                args=(package_url, local_package_file, completed_event, info_queue))
        download_process.start()
        allowed_block_download_time = 60
        download_finished = False
        download_time = time.time()
        while True:
            try:
                queue_time = info_queue.get(block=False, timeout=3)
                download_time = queue_time
            except queue.Empty as e:
                pass

            block_time = time.time() - download_time
            if block_time > allowed_block_download_time:
                break

            if completed_event.is_set():
                download_finished = True
                break
            time.sleep(3)
        try:
            if not download_finished:
                download_process.terminate()
                download_process.kill()
        except Exception as e:
            pass

        if not download_finished:
            raise Exception("Download timeout, please check if your network is stable.")

        if not os.path.exists(local_package_file):
            raise Exception(f"Failed to download, the zip file is not exist at {local_package_file}.")

        # Another method to async download.
        # import socket
        # socket.setdefaulttimeout(15)
        # try:
        #     urllib.request.urlretrieve(package_url, local_package_file,
        #                                reporthook=self.package_download_progress)
        # except socket.timeout:
        #     retry_count = 1
        #     max_retry_num = 5
        #     while retry_count <= max_retry_num:
        #         try:
        #             urllib.request.urlretrieve(package_url, local_package_file,
        #                                        reporthook=self.package_download_progress)
        #             break
        #         except socket.timeout:
        #             error_info = 'Retry %d time' % retry_count if retry_count == 1 else \
        #                 'Reloading for %d times' % retry_count
        #             logging.info(error_info)
        #             retry_count += 1
        #     if retry_count > max_retry_num:
        #         logging.error("Download failed.")
        #         raise Exception("Download failed")

        unzip_package_path = os.path.join(self.agent_package_unzip_dir,
                                          f"unzip_fedml_run_{self.run_id}_{self.edge_id}_{filename_without_extension}")
        try:
            shutil.rmtree(unzip_package_path, ignore_errors=True)
        except Exception as e:
            logging.error(
                f"Failed to remove directory {unzip_package_path}, Exception: {e}, Traceback: {traceback.format_exc()}")
            pass

        # Using unzipped folder name
        package_dir_name = FedMLSchedulerBaseJobRunner.unzip_file(local_package_file, unzip_package_path)
        unzip_package_full_path = os.path.join(unzip_package_path, package_dir_name)

        logging.info("local_package_file {}, unzip_package_path {}, unzip file full path {}".format(
            local_package_file, unzip_package_path, unzip_package_full_path))

        return unzip_package_full_path

    @abstractmethod
    def get_download_package_info(self, packages_config=None):
        download_package_name = packages_config.get("server", None) if self.is_master_runner \
            else packages_config["linuxClient"]
        download_package_url = packages_config.get("serverUrl", None) if self.is_master_runner \
            else packages_config["linuxClientUrl"]
        return download_package_name, download_package_url

    def update_local_fedml_config(self, run_id, run_config):
        # Download the package
        packages_config = run_config["packages_config"]
        download_package_name, download_package_url = self.get_download_package_info(packages_config)
        unzip_package_path = self.retrieve_and_unzip_package(download_package_name, download_package_url)
        fedml_local_config_file = os.path.join(unzip_package_path, "conf", "fedml.yaml")

        # Load the config file to memory
        config_from_container = load_yaml_config(fedml_local_config_file)
        container_entry_file_config = config_from_container["entry_config"]
        container_dynamic_args_config = config_from_container["dynamic_args"]
        entry_file = container_entry_file_config["entry_file"]
        conf_file = container_entry_file_config["conf_file"]
        self.package_type = container_entry_file_config.get("package_type", SchedulerConstants.JOB_PACKAGE_TYPE_DEFAULT)
        full_conf_path = os.path.join(unzip_package_path, "fedml", "config", os.path.basename(conf_file))

        # Dynamically build constrain variable with realtime parameters from server
        self.build_dynamic_constrain_variables(run_id, run_config)

        # Update entry arguments value with constrain variable values with realtime parameters from server
        # currently we support the following constrain variables:
        # ${FEDSYS_RUN_ID}: a run id represented one entire Federated Learning flow
        # ${FEDSYS_PRIVATE_LOCAL_DATA}: private local data path in the Federated Learning client
        # ${FEDSYS_CLIENT_ID_LIST}: client list in one entire Federated Learning flow
        # ${FEDSYS_SYNTHETIC_DATA_URL}: synthetic data url from server,
        #                  if this value is not null, the client will download data from this URL to use it as
        #                  federated training data set
        # ${FEDSYS_IS_USING_LOCAL_DATA}: whether we use private local data as federated training data set
        # container_dynamic_args_config["data_cache_dir"] = "${FEDSYS.PRIVATE_LOCAL_DATA}"
        for constrain_variable_key, constrain_variable_value in self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES.items():
            for argument_key, argument_value in container_dynamic_args_config.items():
                if argument_value is not None and str(argument_value).find(constrain_variable_key) == 0:
                    replaced_argument_value = str(argument_value).replace(
                        constrain_variable_key, str(constrain_variable_value)
                    )
                    container_dynamic_args_config[argument_key] = replaced_argument_value

        # Merge all container new config sections as new config dictionary
        package_conf_object = dict()
        package_conf_object["entry_config"] = container_entry_file_config
        package_conf_object["dynamic_args"] = container_dynamic_args_config
        package_conf_object["dynamic_args"]["config_version"] = self.args.config_version
        container_dynamic_args_config["mqtt_config_path"] = os.path.join(
            unzip_package_path, "fedml", "config", os.path.basename(container_dynamic_args_config["mqtt_config_path"])
        )
        container_dynamic_args_config["s3_config_path"] = os.path.join(
            unzip_package_path, "fedml", "config", os.path.basename(container_dynamic_args_config["s3_config_path"])
        )
        log_file_dir = self.agent_log_file_dir
        os.makedirs(log_file_dir, exist_ok=True)
        package_conf_object["dynamic_args"]["log_file_dir"] = log_file_dir

        # Save new config dictionary to local file
        fedml_updated_config_file = os.path.join(unzip_package_path, "conf", "fedml.yaml")
        GeneralConstants.generate_yaml_doc(package_conf_object, fedml_updated_config_file)

        # Build dynamic arguments and set arguments to fedml config object
        self.build_dynamic_args(run_id, run_config, package_conf_object, unzip_package_path)

        return unzip_package_path, package_conf_object

    def build_dynamic_args(self, run_id, run_config, package_conf_object, base_dir):
        fedml_conf_file = package_conf_object["entry_config"]["conf_file"]
        fedml_conf_file_processed = str(fedml_conf_file).replace('\\', os.sep).replace('/', os.sep)
        fedml_conf_path = os.path.join(base_dir, "fedml", "config",
                                       os.path.basename(fedml_conf_file_processed))
        fedml_conf_object = load_yaml_config(fedml_conf_path)
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})

        # Replace local fedml config objects with parameters from MLOps web
        parameters_object = run_config.get("parameters", None)
        if parameters_object is not None:
            for config_k, config_v in fedml_conf_object.items():
                parameter_v = parameters_object.get(config_k, None)
                if parameter_v is not None:
                    fedml_conf_object[config_k] = parameter_v
                    parameters_object.pop(config_k)

            for config_k, config_v in parameters_object.items():
                fedml_conf_object[config_k] = config_v

        package_dynamic_args = package_conf_object["dynamic_args"]
        if fedml_conf_object.get("comm_args", None) is not None:
            fedml_conf_object["comm_args"]["mqtt_config_path"] = package_dynamic_args["mqtt_config_path"]
            fedml_conf_object["comm_args"]["s3_config_path"] = package_dynamic_args["s3_config_path"]
            fedml_conf_object["common_args"]["using_mlops"] = True
        if fedml_conf_object.get("train_args", None) is not None:
            fedml_conf_object["train_args"]["run_id"] = package_dynamic_args["run_id"]
            fedml_conf_object["train_args"]["client_id_list"] = package_dynamic_args["client_id_list"]
            fedml_conf_object["train_args"]["client_num_in_total"] = int(package_dynamic_args["client_num_in_total"])
            fedml_conf_object["train_args"]["client_num_per_round"] = int(package_dynamic_args["client_num_in_total"])
            fedml_conf_object["train_args"]["client_id"] = self.edge_id
            fedml_conf_object["train_args"]["server_id"] = self.request_json.get("server_id", "0")
        if fedml_conf_object.get("device_args", None) is not None:
            fedml_conf_object["device_args"]["worker_num"] = int(package_dynamic_args["client_num_in_total"])
        # fedml_conf_object["data_args"]["data_cache_dir"] = package_dynamic_args["data_cache_dir"]
        data_args = fedml_conf_object.get("data_args")
        if data_args is not None:
            data_cache_dir = fedml_conf_object["data_args"].get("data_cache_dir")
            if data_cache_dir is not None:
                data_cache_dir = os.path.join(data_cache_dir, str(self.edge_id))
                fedml_conf_object["data_args"]["data_cache_dir"] = data_cache_dir
        if fedml_conf_object.get("tracking_args", None) is not None:
            fedml_conf_object["tracking_args"]["log_file_dir"] = package_dynamic_args["log_file_dir"]
            fedml_conf_object["tracking_args"]["log_server_url"] = package_dynamic_args["log_server_url"]

        fedml_conf_object["dynamic_args"] = package_dynamic_args
        self.fedml_config_object = fedml_conf_object.copy()
        GeneralConstants.generate_yaml_doc(fedml_conf_object, fedml_conf_path)

    def callback_run_bootstrap(self, job_pid):
        GeneralConstants.save_bootstrap_process(self.run_id, job_pid, data_dir=self.agent_data_dir)

    def run_bootstrap_script(self, bootstrap_cmd_list, bootstrap_script_file):
        try:
            logging.info("Bootstrap commands are being executed...")
            process, error_list = GeneralConstants.execute_commands_with_live_logs(
                bootstrap_cmd_list, callback=self.callback_run_bootstrap)

            ret_code, out, err = process.returncode, None, None
            if ret_code is None or ret_code <= 0:
                if error_list is not None and len(error_list) > 0:
                    is_bootstrap_run_ok = False
                else:
                    if out is not None:
                        out_str = sys_utils.decode_our_err_result(out)
                        if out_str != "":
                            logging.info("{}".format(out_str))

                    sys_utils.log_return_info(bootstrap_script_file, 0)

                    is_bootstrap_run_ok = True
            else:
                if err is not None:
                    err_str = sys_utils.decode_our_err_result(err)
                    if err_str != "":
                        logging.error("{}".format(err_str))

                sys_utils.log_return_info(bootstrap_script_file, ret_code)

                is_bootstrap_run_ok = False
        except Exception as e:
            logging.error(f"Bootstrap script error: Exception: {e}, Traceback: {traceback.format_exc()}")
            is_bootstrap_run_ok = False
        return is_bootstrap_run_ok

    def check_runner_stop_event(self):
        if self.run_process_event.is_set():
            logging.info("Received stopping event.")
            raise RunnerError("Runner stopped")

        if self.run_process_completed_event.is_set():
            logging.info("Received completed event.")
            raise RunnerCompletedError("Runner completed")

    def trigger_stop_event(self):
        if self.run_process_event is not None:
            self.run_process_event.set()

        time.sleep(1)
        MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)

    def trigger_completed_event(self):
        if self.run_process_completed_event is not None:
            self.run_process_completed_event.set()

        time.sleep(1)
        MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)

    def execute_job_task(self, unzip_package_path, entry_file_full_path, conf_file_full_path, dynamic_args_config,
                         fedml_config_object):
        run_config = self.request_json["run_config"]
        run_params = run_config.get("parameters", {})
        client_rank = self.request_json.get("client_rank", 1)
        job_yaml = run_params.get("job_yaml", {})
        job_yaml_default_none = run_params.get("job_yaml", None)
        job_api_key = job_yaml.get("run_api_key", None)
        job_api_key = job_yaml.get("fedml_run_dynamic_params", None) if job_api_key is None else job_api_key
        assigned_gpu_ids = run_params.get("gpu_ids", None)
        job_type = job_yaml.get("job_type", None)
        containerize = fedml_config_object.get("containerize", None)
        image_pull_policy = fedml_config_object.get("image_pull_policy", Constants.IMAGE_PULL_POLICY_ALWAYS)
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        conf_file_object = load_yaml_config(conf_file_full_path)
        entry_args_dict = conf_file_object.get("fedml_entry_args", {})
        entry_args = entry_args_dict.get("arg_items", None)
        scheduler_match_info = self.request_json.get("scheduler_match_info", {})
        if job_type == Constants.JOB_TASK_TYPE_TRAIN:
            containerize = True if containerize is None else containerize

        # Bootstrap Info
        bootstrap_script_path, bootstrap_script_dir, bootstrap_script_file = [None] * 3
        env_args = fedml_config_object.get("environment_args", None)

        if env_args is not None:
            bootstrap_script_file = env_args.get("bootstrap", None)
            if bootstrap_script_file is not None:
                bootstrap_script_file = str(bootstrap_script_file).replace('\\', os.sep).replace('/', os.sep)
                if platform.system() == 'Windows':
                    bootstrap_script_file = bootstrap_script_file.rstrip('.sh') + '.bat'
                if bootstrap_script_file is not None:
                    bootstrap_script_dir = os.path.join(unzip_package_path, "fedml",
                                                        os.path.dirname(bootstrap_script_file))
                    bootstrap_script_path = os.path.join(
                        bootstrap_script_dir, bootstrap_script_dir, os.path.basename(bootstrap_script_file)
                    )

        bootstrap_cmd_list = list()
        if bootstrap_script_path:
            logging.info("Bootstrap commands are being generated...")
            bootstrap_cmd_list = JobRunnerUtils.generate_bootstrap_commands(bootstrap_script_path=bootstrap_script_path,
                                                                            bootstrap_script_dir=bootstrap_script_dir,
                                                                            bootstrap_script_file=bootstrap_script_file)
            logging.info(f"Generated following Bootstrap commands: {bootstrap_cmd_list}")

        if not containerize:
            if len(bootstrap_cmd_list) and not (job_type == Constants.JOB_TASK_TYPE_DEPLOY or
                                                job_type == Constants.JOB_TASK_TYPE_SERVE):
                bootstrapping_successful = self.run_bootstrap_script(bootstrap_cmd_list=bootstrap_cmd_list,
                                                                     bootstrap_script_file=bootstrap_script_file)

                if not bootstrapping_successful:
                    logging.info("failed to update local fedml config.")
                    self.check_runner_stop_event()
                    # Send failed msg when exceptions.
                    raise Exception(f"Failed to execute following bootstrap commands: {bootstrap_cmd_list}")

                logging.info("cleanup the previous learning process and bootstrap process...")
                GeneralConstants.cleanup_learning_process(self.request_json["runId"], data_dir=self.agent_data_dir)
                GeneralConstants.cleanup_bootstrap_process(self.request_json["runId"], data_dir=self.agent_data_dir)

        executable_interpreter = GeneralConstants.CLIENT_SHELL_PS \
            if platform.system() == GeneralConstants.PLATFORM_WINDOWS else GeneralConstants.CLIENT_SHELL_BASH

        if job_yaml_default_none is None:
            # Generate the job executing commands for previous federated learning (Compatibility)
            python_program = get_python_program()
            rank = str(dynamic_args_config.get("rank", 1))
            role = "server" if rank == "0" else "client"
            logging.info(f"Run the {role}: {python_program} {entry_file_full_path} --cf {conf_file_full_path} "
                         f"--rank {rank} --role {role}")
            entry_command = f"{python_program} {entry_file_full_path} --cf " \
                            f"{conf_file_full_path} --rank {rank} --role {role}"
            shell_cmd_list = [entry_command]

            # Run the job executing commands for previous federated learning (Compatibility)
            process, error_list = GeneralConstants.execute_commands_with_live_logs(
                shell_cmd_list, callback=self.callback_start_fl_job, should_write_log_file=False)
            is_launch_task = False
        else:
            self.check_runner_stop_event()

            if self.is_master_runner:
                self.status_reporter.report_server_id_status(
                    self.run_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_RUNNING, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.edge_id)
            else:
                self.status_reporter.report_client_id_status(
                    self.edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_RUNNING, run_id=self.run_id)

            # Generate the job executing commands
            job_executing_commands = JobRunnerUtils.generate_job_execute_commands(
                self.run_id, self.edge_id, self.version,
                self.package_type, executable_interpreter, entry_file_full_path,
                conf_file_object, entry_args, assigned_gpu_ids,
                job_api_key, client_rank, scheduler_match_info=scheduler_match_info,
                cuda_visible_gpu_ids_str=self.cuda_visible_gpu_ids_str)

            if containerize is not None and containerize is True:
                docker_args = fedml_config_object.get("docker", {})
                docker_args = JobRunnerUtils.create_instance_from_dict(DockerArgs, docker_args)
                try:
                    job_executing_commands = JobRunnerUtils.generate_launch_docker_command(
                        docker_args=docker_args,  run_id=self.run_id, edge_id=self.edge_id,
                        unzip_package_path=unzip_package_path, executable_interpreter=executable_interpreter,
                        entry_file_full_path=entry_file_full_path, bootstrap_cmd_list=bootstrap_cmd_list,
                        cuda_visible_gpu_ids_str=self.cuda_visible_gpu_ids_str, image_pull_policy=image_pull_policy)
                except Exception as e:
                    logging.error(f"Error occurred while generating containerized launch commands. "
                                  f"Exception: {e}, Traceback: {traceback.format_exc()}")
                    return None, None, None

                if not job_executing_commands:
                    raise Exception("Failed to generate docker execution command")

            # Run the job executing commands
            logging.info(f"Run the client job with job id {self.run_id}, device id {self.edge_id}.")
            process, error_list = GeneralConstants.execute_commands_with_live_logs(
                job_executing_commands, callback=self.start_job_perf, error_processor=self.job_error_processor,
                should_write_log_file=False if job_type == Constants.JOB_TASK_TYPE_FEDERATE else True)
            is_launch_task = False if job_type == Constants.JOB_TASK_TYPE_FEDERATE else True

        return process, is_launch_task, error_list

    def callback_start_fl_job(self, job_pid):
        GeneralConstants.save_learning_process(self.run_id, job_pid, data_dir=self.agent_data_dir)
        self.mlops_metrics.report_sys_perf(
            self.args, self.agent_config["mqtt_config"], job_process_id=job_pid)

    def start_job_perf(self, job_pid):
        GeneralConstants.save_learning_process(self.run_id, job_pid, data_dir=self.agent_data_dir)
        #self.mlops_metrics.report_job_perf(self.args, self.agent_config["mqtt_config"], job_pid)

    def job_error_processor(self, error_list):
        self.check_runner_stop_event()

        error_str = "\n".join(error_list)
        error_message = f"Error occurred when running the job... {error_str}"
        logging.error(error_message)
        raise Exception(error_message)

    def start_runner_process(
            self, run_id, edge_id, request_json,  cuda_visible_gpu_ids_str=None,
            sender_message_queue=None, listener_message_queue=None,
            status_center_queue=None, process_name=None
    ):
        return None

    @staticmethod
    def cleanup_containers_and_release_gpus(run_id, edge_id, job_type=SchedulerConstants.JOB_TASK_TYPE_TRAIN):
        # Check if the job type is not "serve" or "deploy"
        if not (job_type == SchedulerConstants.JOB_TASK_TYPE_SERVE or
                job_type == SchedulerConstants.JOB_TASK_TYPE_DEPLOY):

            # Terminate the run docker container if exists
            try:
                container_name = JobRunnerUtils.get_run_container_name(run_id)
                docker_client = JobRunnerUtils.get_docker_client(DockerArgs())
                logging.info(f"Terminating the run docker container {container_name} if exists...")
                JobRunnerUtils.remove_run_container_if_exists(container_name, docker_client)
            except Exception as e:
                logging.error(f"Exception {e} occurred when terminating docker container. "
                              f"Traceback: {traceback.format_exc()}")

            # Release the GPU ids and update the GPU availability in the persistent store
            JobRunnerUtils.get_instance().release_gpu_ids(run_id, edge_id)

            # Send mqtt message reporting the new gpu availability to the backend
            MLOpsDevicePerfStats.report_gpu_device_info(edge_id)

    def rebuild_message_status_center(self, sender_message_queue, listener_message_queue, status_queue):
        self.message_center = FedMLMessageCenter.rebuild_message_center_from_queue(
            sender_message_queue, listener_message_queue=listener_message_queue)
        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self.message_center)
        self.mlops_metrics.run_id = self.run_id

        self.status_center = FedMLStatusCenter.rebuild_status_center_from_queue(status_queue)
        if self.status_reporter is None:
            self.status_reporter = MLOpsMetrics()
        self.status_reporter.set_messenger(self.status_center)
        self.status_reporter.run_id = self.run_id
