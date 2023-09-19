import os
import platform
import signal

import psutil
import yaml
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config


class RunProcessUtils:
    @staticmethod
    def get_run_process_prefix(prefix, run_id):
        return f"{prefix}-run@{run_id}@pid@"

    @staticmethod
    def cleanup_run_process(run_id, data_dir, info_dir, info_file_prefix="runner-process"):
        try:
            local_pkg_data_dir = data_dir
            run_process_dir = os.path.join(local_pkg_data_dir, info_dir)
            run_process_files = os.listdir(run_process_dir)
            for process_file in run_process_files:
                if run_id is None:
                    run_splits = process_file.split("@")
                    process_id = None if len(run_splits) < 4 else run_splits[3]
                else:
                    run_pid_prefix = RunProcessUtils.get_run_process_prefix(info_file_prefix, run_id)
                    if not process_file.startswith(run_pid_prefix):
                        continue

                    process_id = process_file.split(run_pid_prefix)
                    if process_id is None or process_id == "":
                        continue

                try:
                    process = psutil.Process(process_id)
                    child_processes = process.children(recursive=True)
                    for sub_process in child_processes:
                        if platform.system() == 'Windows':
                            os.system("taskkill /PID {} /T /F".format(sub_process.pid))
                        else:
                            os.kill(sub_process.pid, signal.SIGKILL)

                    if process is not None:
                        if platform.system() == 'Windows':
                            os.system("taskkill /PID {} /T /F".format(process.pid))
                        else:
                            os.killpg(os.getpgid(process_id), signal.SIGKILL)
                except Exception as e:
                    pass

                try:
                    os.remove(os.path.join(run_process_dir, process_file))
                except Exception as e:
                    pass

        except Exception as e:
            pass

    @staticmethod
    def save_run_process(run_id, process_id, data_dir, info_dir,
                         process_info=None, info_file_prefix="runner-process"):
        try:
            local_pkg_data_dir = data_dir
            process_id_file = os.path.join(local_pkg_data_dir, info_dir,
                                           f"{RunProcessUtils.get_run_process_prefix(info_file_prefix, run_id)}"
                                           f"{process_id}")
            if os.path.exists(process_id_file):
                process_info_dict = load_yaml_config(process_id_file)
            else:
                process_info_dict = dict()
            process_info_dict["info"] = process_info if process_info is not None else f"run-{run_id}-pid-{process_id}"
            process_info_dict["run_id"] = run_id
            process_info_dict["pid"] = process_id
            RunProcessUtils.generate_yaml_doc(process_info_dict, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def generate_yaml_doc(run_config_object, yaml_file):
        try:
            file = open(yaml_file, 'w', encoding='utf-8')
            yaml.dump(run_config_object, file)
            file.close()
        except Exception as e:
            pass