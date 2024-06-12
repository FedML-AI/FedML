import logging
import os
import platform
import signal
import uuid
from os.path import expanduser
from typing import List

import chardet
import psutil
import yaml

from fedml.computing.scheduler.comm_utils.hardware_utils import HardwareUtil
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
import json
from urllib import request
from pkg_resources import parse_version
import fedml
from packaging import version
import sys
import subprocess

from fedml.computing.scheduler.slave.client_constants import ClientConstants

FETAL_ERROR_START_CODE = 128

SYS_ERR_CODE_MAP = {"0": "Successful exit without errors.",
                    "1": "One or more generic errors encountered upon exit.",
                    "2": "Incorrect usage, such as invalid options or missing arguments.",
                    "126": "Command found but is not executable.",
                    "127": "Command not found, usually the result of a missing directory in PATH variable.",
                    "128": "Command encountered fatal error "
                           "(was forcefully terminated manually or from an outside source).",
                    "130": "Command terminated with signal 2 (SIGINT) (ctrl+c on keyboard).",
                    "143": "Command terminated with signal 15 (SIGTERM) (kill command)."}

enable_simulation_gpu = False
simulation_gpu_count = 8


def get_sys_runner_info():
    import fedml
    fedml_ver = str(fedml.__version__)
    exec_path = str(os.path.abspath(fedml.__file__))
    os_ver = ""
    cpu_info = ""
    python_ver = ""
    torch_ver = ""
    mpi_installed = False
    cpu_usage = 0.0
    available_mem = ""
    total_mem = ""
    gpu_info = None
    gpu_available_mem = None
    gpu_total_mem = None
    gpu_count = 0
    gpu_vendor = None
    cpu_count = 1
    gpu_device_name = None

    import platform
    os_ver = platform.platform()
    cpu_info = platform.machine()

    import sys
    python_ver = sys.version

    try:
        import torch

        torch_ver = torch.__version__
    except:
        torch_ver = ""
        pass

    try:
        from mpi4py import MPI
        mpi_obj = mpi4py.MPI
        mpi_installed = True
    except:
        pass

    try:
        import psutil

        # Getting loadover15 minutes
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = "{:.0f}%".format((load15 / os.cpu_count()) * 100)
        available_mem = "{:.1f} G".format(psutil.virtual_memory().available / 1024 / 1024 / 1024)
        total_mem = "{:.1f}G".format(psutil.virtual_memory().total / 1024 / 1024 / 1024)
    except:
        cpu_usage = ""
        available_mem = ""
        total_mem = ""
        pass

    try:
        gpus = HardwareUtil.get_gpus()
        memory_total = 0.0
        memory_free = 0.0
        for gpu in gpus:
            memory_total += gpu.memoryTotal
            memory_free += gpu.memoryFree

        gpu_available_mem = "{:.1f} G".format(memory_free / 1024.0)
        gpu_total_mem = "{:.1f}G".format(memory_total / 1024.0)
        gpu_count = len(gpus)
        if gpu_count:
            gpu_vendor = gpus[0].vendor
            gpu_device_name = gpus[0].name

        # gpu_device_name = torch.cuda.get_device_name(0)
        gpu_info = gpu_device_name
    except:
        pass

    enable_simulation_gpu, simulation_gpu_count = get_simulation_gpu_env()

    if enable_simulation_gpu:
        gpu_count = simulation_gpu_count
        gpu_total_mem = "80G"
        gpu_available_mem = "80G"
        gpu_vendor = "NVIDIA"
        gpu_device_name = "NVIDIA A100-SXM4-80GB"

    cpu_count = os.cpu_count()

    return fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
        cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem, \
        gpu_count, gpu_vendor, cpu_count, gpu_device_name


def get_simulation_gpu_env():
    _enable_simulation_gpu = enable_simulation_gpu
    _simulation_gpu_count = simulation_gpu_count

    env_enable_simulation_gpu = os.getenv("FEDML_ENABLE_SIMULATION_GPU", None)
    if env_enable_simulation_gpu is not None:
        _enable_simulation_gpu = True if env_enable_simulation_gpu == "1" or env_enable_simulation_gpu == 1 else False

    env_simulation_gpu_count = os.getenv("FEDML_SIMULATION_GPU_COUNT", None)
    if env_simulation_gpu_count is not None:
        _simulation_gpu_count = int(env_simulation_gpu_count)

    return _enable_simulation_gpu, _simulation_gpu_count


# GPU list: [GPU(ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver,
# gpu_name, serial, display_mode, display_active, temperature)]
def get_gpu_list():
    enable_simulation_gpu, simulation_gpu_count = get_simulation_gpu_env()

    if enable_simulation_gpu:
        ret_gpu_list = [
            {'ID': 0, 'uuid': 'GPU-dab987f0-be09-294a-96d6-f9afeef49877', 'load': 1.0,
             'memoryTotal': 81920.0, 'memoryUsed': 41738.0, 'memoryFree': 39311.0,
             'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB', 'serial': '1320723001415',
             'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 48.0},
            {'ID': 1, 'uuid': 'GPU-862efda8-4d58-3a46-31f8-ce3a271aae43', 'load': 0.0, 'memoryTotal': 81920.0,
             'memoryUsed': 7.0, 'memoryFree': 81042.0, 'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
             'serial': '1320723002357', 'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 29.0},
            {'ID': 2, 'uuid': 'GPU-c2e4df14-5055-1bc7-54f8-54f063621272', 'load': 0.0, 'memoryTotal': 81920.0,
             'memoryUsed': 7.0, 'memoryFree': 81042.0, 'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
             'serial': '1320723001450', 'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 29.0},
            {'ID': 3, 'uuid': 'GPU-d60909cf-6436-2aaa-d162-9eb66bbe301c', 'load': 0.0, 'memoryTotal': 81920.0,
             'memoryUsed': 7.0, 'memoryFree': 81042.0, 'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
             'serial': '1320723007910', 'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 30.0},
            {'ID': 4, 'uuid': 'GPU-54345e96-e7c4-6b7e-0330-eebe4e1ef139', 'load': 0.0, 'memoryTotal': 81920.0,
             'memoryUsed': 7.0, 'memoryFree': 81042.0, 'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
             'serial': '1320723001651', 'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 30.0},
            {'ID': 5, 'uuid': 'GPU-dc6938d5-d1e6-fe54-1244-13ee4045687f', 'load': 0.0, 'memoryTotal': 81920.0,
             'memoryUsed': 7.0, 'memoryFree': 81042.0, 'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
             'serial': '1320723005960', 'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 28.0},
            {'ID': 6, 'uuid': 'GPU-5248df36-489a-62ae-b787-6b21a0f8e388', 'load': 0.0, 'memoryTotal': 81920.0,
             'memoryUsed': 7.0, 'memoryFree': 81042.0, 'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
             'serial': '1320723004499', 'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 31.0},
            {'ID': 7, 'uuid': 'GPU-b5811fb0-e93a-79c7-1548-2d2b60049207', 'load': 0.0, 'memoryTotal': 81920.0,
             'memoryUsed': 7.0, 'memoryFree': 81042.0, 'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
             'serial': '1320723000504', 'display_mode': 'Enabled', 'display_active': 'Disabled', 'temperature': 33.0}]

        if simulation_gpu_count > 8:
            for count in range(8, simulation_gpu_count):
                ret_gpu_list.append(
                    {'ID': count, 'uuid': f"GPU-b5811fb0-e93a-79c7-1548-2d2b60049208{count}", 'load': 0.0,
                     'memoryTotal': 81920.0, 'memoryUsed': 7.0, 'memoryFree': 81042.0,
                     'driver': '535.54.03', 'gpu_name': 'NVIDIA A100-SXM4-80GB',
                     'serial': f"1320723000504{count}", 'display_mode': 'Enabled',
                     'display_active': 'Disabled', 'temperature': 33.0})

        return ret_gpu_list[0:simulation_gpu_count]

    gpu_list = HardwareUtil.get_gpus()
    ret_gpu_list = list()
    for gpu in gpu_list:
        ret_gpu_item = {"ID": gpu.id, "uuid": gpu.uuid, "load": gpu.load,
                        "memoryTotal": gpu.memoryTotal, "memoryUsed": gpu.memoryUsed,
                        "memoryFree": gpu.memoryFree, "driver": gpu.driver,
                        "gpu_name": gpu.name, "serial": gpu.serial,
                        "display_mode": gpu.display_mode, "display_active": gpu.display_active,
                        "temperature": gpu.temperature}
        ret_gpu_list.append(ret_gpu_item)
    return ret_gpu_list


def get_available_gpu_id_list(limit=1) -> List[int]:
    enable_simulation_gpu, simulation_gpu_count = get_simulation_gpu_env()

    if enable_simulation_gpu:
        available_gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        if simulation_gpu_count > 8:
            for count in range(8, simulation_gpu_count):
                available_gpu_ids.append(count)
        return available_gpu_ids[0:simulation_gpu_count]

    gpu_available_list = HardwareUtil.get_available_gpu_ids(order='memory', limit=limit, max_load=0.01,
                                                            max_memory=0.01)
    return gpu_available_list


def get_host_name():
    host_name = None
    try:
        import platform
        host_name = platform.uname()[1]
    except Exception as e:
        pass
    return host_name


def generate_yaml_doc(yaml_object, yaml_file, append=False):
    try:
        open_mode = "w"
        if append:
            open_mode = "a"
        file = open(yaml_file, open_mode, encoding="utf-8")
        yaml.dump(yaml_object, file)
        file.close()
    except Exception as e:
        pass


def get_gpu_count_vendor():
    gpu_count = 0
    gpu_vendor = ""
    try:
        gpus = HardwareUtil.get_gpus()
        gpu_count = len(gpus)
        if gpu_count:
            gpu_vendor = gpus[0].vendor
    except:
        pass

    return gpu_count, gpu_vendor


def get_running_info(cs_home_dir, cs_info_dir):
    home_dir = expanduser("~")
    runner_info_file = os.path.join(
        home_dir, ".fedml", cs_home_dir, "fedml", "data", cs_info_dir, "runner_infos.yaml"
    )
    if os.path.exists(runner_info_file):
        running_info = load_yaml_config(runner_info_file)
        return running_info["run_id"], running_info["edge_id"]
    return 0, 0


def get_python_program():
    python_program = "python3"
    current_python_version = sys.version.split(" ")[0]
    try:
        python_version_str = os.popen("python --version").read()
        if python_version_str.find(current_python_version) != -1:
            python_program = "python"
        else:
            python3_version_str = os.popen("python3 --version").read()
            if python3_version_str.find(current_python_version) != -1:
                python_program = "python3"
    except Exception as e:
        pass

    return python_program


def cleanup_login_process(runner_home_dir, runner_info_dir):
    try:
        home_dir = expanduser("~")
        local_pkg_data_dir = os.path.join(home_dir, ".fedml", runner_home_dir, "fedml", "data")
        edge_process_id_file = os.path.join(
            local_pkg_data_dir, runner_info_dir, "runner-process.id"
        )
        edge_process_info = load_yaml_config(edge_process_id_file)
        edge_process_id = edge_process_info.get("process_id", None)
        if edge_process_id is not None:
            edge_process = psutil.Process(edge_process_id)
            if edge_process is not None:
                if platform.system() == 'Windows':
                    os.system("taskkill /PID {} /T /F".format(edge_process.pid))
                else:
                    os.killpg(os.getpgid(edge_process.pid), signal.SIGKILL)
        yaml_object = {}
        yaml_object["process_id"] = -1
        generate_yaml_doc(yaml_object, edge_process_id_file)

    except Exception as e:
        pass


def save_login_process(runner_home_dir, runner_info_dir, edge_process_id):
    home_dir = expanduser("~")
    local_pkg_data_dir = os.path.join(home_dir, ".fedml", runner_home_dir, "fedml", "data")
    os.makedirs(local_pkg_data_dir, exist_ok=True)
    os.makedirs(os.path.join(local_pkg_data_dir, runner_info_dir), exist_ok=True)

    try:
        edge_process_id_file = os.path.join(
            local_pkg_data_dir, runner_info_dir, "runner-process.id"
        )
        yaml_object = {}
        yaml_object["process_id"] = edge_process_id
        generate_yaml_doc(yaml_object, edge_process_id_file)
    except Exception as e:
        pass


def cleanup_all_fedml_client_learning_processes():
    # Cleanup all fedml client learning processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            found_learning_process = False
            found_client_process = False
            for cmd in pinfo["cmdline"]:
                if str(cmd).find("fedml_config.yaml") != -1:
                    found_learning_process = True

                if str(cmd).find("client") != -1:
                    found_client_process = True

            if found_learning_process and found_client_process:
                # click.echo("find client learning process at {}.".format(process.pid))
                if platform.system() == 'Windows':
                    os.system("taskkill /PID {} /T /F".format(process.pid))
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_all_fedml_client_diagnosis_processes():
    # Cleanup all fedml client learning processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            found_client_diagnosis_process = False
            for cmd in pinfo["cmdline"]:
                if str(cmd).find("client_diagnosis") != -1:
                    found_client_diagnosis_process = True

            if found_client_diagnosis_process:
                # click.echo("find client diagnosis process at {}.".format(process.pid))
                if platform.system() == 'Windows':
                    os.system("taskkill /PID {} /T /F".format(process.pid))
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_all_fedml_client_login_processes(login_program, clean_process_group=True):
    # Cleanup all fedml client login processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            for cmd in pinfo["cmdline"]:
                if str(cmd).find(login_program) != -1:
                    if os.path.basename(cmd) == login_program:
                        # click.echo("find client login process at {}.".format(process.pid))
                        if platform.system() == "Windows":
                            os.system("taskkill /PID {} /T /F".format(process.pid))
                        else:
                            os.kill(process.pid, signal.SIGKILL)
                            if clean_process_group:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_all_fedml_server_learning_processes():
    # Cleanup all fedml server learning processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            found_learning_process = False
            found_server_process = False
            for cmd in pinfo["cmdline"]:
                if str(cmd).find("fedml_config.yaml") != -1:
                    found_learning_process = True

                if str(cmd).find("server") != -1:
                    found_server_process = True

            if found_learning_process and found_server_process:
                # click.echo("find server learning process at {}.".format(process.pid))
                if platform.system() == 'Windows':
                    os.system("taskkill /PID {} /T /F".format(process.pid))
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_all_fedml_client_api_processes(kill_all=False, is_model_device=False):
    # Cleanup all fedml client api processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            find_api_process = False
            for cmd in pinfo["cmdline"]:
                if is_model_device:
                    if str(cmd).find("model_scheduler.device_client_api:api") != -1:
                        find_api_process = True
                else:
                    if str(cmd).find("slave.client_api:api") != -1:
                        find_api_process = True

            if find_api_process:
                # click.echo("find client api process at {}.".format(process.pid))
                if platform.system() == 'Windows':
                    os.system("taskkill /PID {} /T /F".format(process.pid))
                else:
                    if kill_all:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        os.kill(process.pid, signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_all_fedml_server_api_processes(kill_all=False, is_model_device=False):
    # Cleanup all fedml server api processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            find_api_process = False
            for cmd in pinfo["cmdline"]:
                if is_model_device:
                    if str(cmd).find("model_scheduler.device_server_api:api") != -1:
                        find_api_process = True

                    if str(cmd).find("model_scheduler.device_model_inference:api") != -1:
                        find_api_process = True
                else:
                    if str(cmd).find("master.server_api:api") != -1:
                        find_api_process = True

            if find_api_process:
                # click.echo("find server api process at {}.".format(process.pid))
                if platform.system() == 'Windows':
                    os.system("taskkill /PID {} /T /F".format(process.pid))
                else:
                    if kill_all:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        os.kill(process.pid, signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_all_fedml_server_login_processes(login_program, clean_process_group=False):
    # Cleanup all fedml client login processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            for cmd in pinfo["cmdline"]:
                if str(cmd).find(login_program) != -1:
                    if os.path.basename(cmd) == login_program:
                        # click.echo("find server login process at {}.".format(process.pid))
                        if platform.system() == 'Windows':
                            os.system("taskkill /PID {} /T /F".format(process.pid))
                        else:
                            os.kill(process.pid, signal.SIGKILL)
                            if clean_process_group:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_all_bootstrap_processes(bootstrap_program, clean_process_group=False):
    # Cleanup all fedml bootstrap processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            for cmd in pinfo["cmdline"]:
                if str(cmd).find(bootstrap_program) != -1:
                    if os.path.basename(cmd) == bootstrap_program:
                        # click.echo("find server login process at {}.".format(process.pid))
                        if platform.system() == 'Windows':
                            os.system("taskkill /PID {} /T /F".format(process.pid))
                        else:
                            os.kill(process.pid, signal.SIGKILL)
                            if clean_process_group:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception as e:
            pass


def cleanup_model_monitor_processes(run_id, end_point_name, model_id, model_name, model_version):
    # Cleanup all fedml server api processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            find_monitor_process = False
            find_monitor_name_arg = False
            find_endpoint_id_name_arg = False
            for cmd in pinfo["cmdline"]:
                if str(cmd).endswith("device_model_monitor.py"):
                    find_monitor_name_arg = True

                if find_monitor_name_arg and str(cmd) == f"-ep":
                    find_endpoint_id_name_arg = True

                if find_monitor_name_arg and find_endpoint_id_name_arg and str(cmd) == f"{run_id}":
                    find_monitor_process = True
                    break

            if find_monitor_process:
                # click.echo("find the monitor process at {}.".format(process.pid))
                if platform.system() == 'Windows':
                    os.system("taskkill /PID {} /T /F".format(process.pid))
                else:
                    os.kill(process.pid, signal.SIGKILL)
                break
        except Exception as e:
            pass


def get_process_running_count(process_name):
    count = 0
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            for cmd in pinfo["cmdline"]:
                if str(cmd).find(process_name) != -1:
                    if os.path.basename(cmd) == process_name:
                        count += 1
        except Exception as e:
            pass

    return count


def edge_simulator_has_login(login_program="client_login.py"):
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            found_login_process = False
            found_simulator_process = False
            for cmd in pinfo["cmdline"]:
                if str(cmd).find(login_program) != -1:
                    if os.path.basename(cmd) == login_program:
                        found_login_process = True

                if str(cmd).find("edge_simulator") != -1:
                    found_simulator_process = True

            if found_login_process and found_simulator_process:
                return True
        except Exception as e:
            pass

    return False


def save_simulator_process(data_dir, runner_info_dir, process_id, run_id, run_status=None):
    simulator_proc_path = os.path.join(data_dir, runner_info_dir, "simulator-processes")
    os.makedirs(simulator_proc_path, exist_ok=True)

    try:
        simulator_process_id_file = os.path.join(
            simulator_proc_path, "simulator-process-{}".format(str(process_id))
        )
        yaml_object = dict()
        yaml_object["run_id"] = str(run_id)
        if run_status is not None:
            yaml_object["run_status"] = run_status
        generate_yaml_doc(yaml_object, simulator_process_id_file, append=False)
    except Exception as e:
        pass


def get_simulator_process_list(data_dir, runner_info_dir):
    simulator_proc_path = os.path.join(data_dir, runner_info_dir, "simulator-processes")
    if os.path.exists(simulator_proc_path):
        process_files = os.listdir(simulator_proc_path)
        running_info = dict()
        status_info = dict()
        for process_file in process_files:
            process_spit = str(process_file).split('-')
            if len(process_spit) == 3:
                process_id = process_spit[2]
            else:
                continue
            run_id_info = load_yaml_config(os.path.join(simulator_proc_path, process_file))
            running_info[str(process_id)] = run_id_info["run_id"]
            status_info[str(run_id_info["run_id"])] = run_id_info.get("run_status", "")

        return running_info, status_info
    else:
        return dict(), dict()


def remove_simulator_process(data_dir, runner_info_dir, process_id):
    simulator_proc_path = os.path.join(data_dir, runner_info_dir, "simulator-processes")
    os.makedirs(simulator_proc_path, exist_ok=True)

    try:
        simulator_process_id_file = os.path.join(
            simulator_proc_path, "simulator-process-{}".format(str(process_id))
        )
        os.remove(simulator_process_id_file)
    except Exception as e:
        pass


def remove_files(file_paths: List[str]):
    """
    Remove files if they exist.

    Args:
        file_paths: List of file paths

    Usage:
        file_list = ["file4.txt", "file5.txt", "file6.txt"]
        remove_files(file_list)
    """
    if not isinstance(file_paths, list):
        raise ValueError("file_paths must be a list of file paths.")

    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)


def convert_and_remove_bat_files(shell_script_paths: List[str]):
    if not isinstance(shell_script_paths, list):
        raise ValueError("sh_file_paths must be a list of file paths.")

    # Convert to bat file paths
    bat_file_paths = list(map(lambda path: path[:-2] + 'bat', shell_script_paths))

    # Filter out non-bat paths
    bat_file_paths = list(filter(lambda path: path.endswith('.bat'), bat_file_paths))

    remove_files(bat_file_paths)


def simulator_process_is_running(process_id):
    for process in psutil.process_iter():
        if str(process.pid) == str(process_id):
            return True

    return False


def log_return_info(bootstrap_file, ret_code):
    import logging
    err_desc = SYS_ERR_CODE_MAP.get(str(ret_code), "")
    if ret_code == 0:
        logging.info("Run '{}' return code {}. {}".format(
            bootstrap_file, ret_code, err_desc))
    else:
        fatal_err_desc = SYS_ERR_CODE_MAP.get(str(ret_code), "")
        if ret_code >= FETAL_ERROR_START_CODE and fatal_err_desc == "":
            fatal_err_desc = SYS_ERR_CODE_MAP.get(str(FETAL_ERROR_START_CODE))

        logging.error("Run '{}' return code {}. {}".format(
            bootstrap_file, ret_code, fatal_err_desc if fatal_err_desc != "" else err_desc))


def get_device_id_in_docker():
    docker_env_file = "/.dockerenv"
    cgroup_file = "/proc/1/cgroup"
    product_uuid_file = "/sys/class/dmi/id/product_uuid"

    if os.path.exists(docker_env_file) or os.path.exists(cgroup_file):
        is_in_docker = False
        try:
            with open(cgroup_file, 'r') as f:
                while True:
                    cgroup_line = f.readline()
                    if len(cgroup_line) <= 0:
                        break
                    name = cgroup_line.find(":name=")
                    devices = cgroup_line.find(":device:")
                    name_docker_res = cgroup_line.find("docker")
                    devices_docker_res = cgroup_line.find("docker")
                    name_pod_res = cgroup_line.find("pod")
                    devices_pod_res = cgroup_line.find("pod")
                    if name != -1 and (name_docker_res != -1 or name_pod_res != -1):
                        is_in_docker = True
                        break
                    if devices != -1 and (devices_docker_res != -1 or devices_pod_res != -1):
                        is_in_docker = True
                        break
        except Exception as e:
            pass

        if os.path.exists(docker_env_file):
            is_in_docker = True

        if not is_in_docker:
            return None

        try:
            with open(product_uuid_file, 'r') as f:
                sys_device_id = f.readline().rstrip("\n").strip(" ")
                if sys_device_id is None or sys_device_id == "":
                    device_id = str(uuid.uuid4())
                else:
                    device_id = "{}-@-{}".format(sys_device_id, str(uuid.uuid4()))
                return f"{device_id}-docker"
        except Exception as e:
            device_id = str(uuid.uuid4())
            return f"{device_id}-docker"
    return None


def versions(configuration_env, pkg_name):
    if configuration_env == "release":
        url = f'https://pypi.python.org/pypi/{pkg_name}/json'
    else:
        url = f'https://test.pypi.org/pypi/{pkg_name}/json'
    import ssl
    context = ssl._create_unverified_context()
    releases = json.loads(request.urlopen(url, context=context).read())['releases']
    return sorted(releases, key=parse_version, reverse=True)


def upgrade_if_not_latest():
    disable_ota = os.getenv("FEDML_LAUNCH_DISABLE_OTA")
    if disable_ota is not None and disable_ota == "YES":
        return

    try:
        config_version = fedml.get_env_version()
        is_latest_version, _, _ = check_fedml_is_latest_version(configuration_env=config_version)
        if not is_latest_version:
            daemon_ota_upgrade_with_version(config_version)
            print("Completed upgrading, please launch your job again.")
            exit(-1)
    except Exception as e:
        pass


# Policy for version checking.
# Dev: keep tracking the latest alpha, beta version
# Test: don't check, keep the local version
# Release: keep tracking the latest release version
def check_fedml_is_latest_version(configuration_env="release"):
    fedml_version_list = versions(configuration_env, "fedml")
    local_fedml_version = fedml.__version__

    local_fedml_ver_info = version.parse(local_fedml_version)
    if configuration_env == "dev":
        # For the dev env, we just check if the version is dev release
        for remote_ver_item in fedml_version_list:
            remote_fedml_ver_info = version.parse(remote_ver_item)
            if not remote_fedml_ver_info.is_devrelease:
                continue

            if local_fedml_ver_info < remote_fedml_ver_info:
                return False, local_fedml_version, remote_ver_item
            else:
                return True, local_fedml_version, remote_ver_item

        return True, local_fedml_version, local_fedml_version
    elif configuration_env == "test":
        # For the dev env, we just check if the version is alpha version of pre release
        for remote_ver_item in fedml_version_list:
            remote_fedml_ver_info = version.parse(remote_ver_item)
            if not remote_fedml_ver_info.is_prerelease or remote_fedml_ver_info.pre[0] != "rc":
                continue

            if local_fedml_ver_info < remote_fedml_ver_info:
                return False, local_fedml_version, remote_ver_item
            else:
                return True, local_fedml_version, remote_ver_item

        return True, local_fedml_version, local_fedml_version
    elif configuration_env == "release":
        # For the dev env, we just check if the version is rc version of pre release, post release and release version
        for remote_ver_item in fedml_version_list:
            remote_fedml_ver_info = version.parse(remote_ver_item)
            if (remote_fedml_ver_info.is_prerelease and remote_fedml_ver_info.pre[0] != "rc") or \
                    (remote_fedml_ver_info.is_devrelease):
                continue

            if local_fedml_ver_info < remote_fedml_ver_info:
                return False, local_fedml_version, remote_ver_item
            else:
                return True, local_fedml_version, remote_ver_item

        return True, local_fedml_version, local_fedml_version
    else:
        return True, local_fedml_version, local_fedml_version


def daemon_ota_upgrade(in_args):
    should_upgrade = False
    fedml_is_latest_version = True
    try:
        fedml_is_latest_version, local_ver, remote_ver = check_fedml_is_latest_version(in_args.version)
        should_upgrade = False if fedml_is_latest_version else True
    except Exception as e:
        return

    if not should_upgrade:
        return
    upgrade_version = remote_ver

    do_upgrade(in_args.version, upgrade_version, show_local_console=True)


def daemon_ota_upgrade_with_version(in_version="release"):
    should_upgrade = False
    fedml_is_latest_version = True
    try:
        fedml_is_latest_version, local_ver, remote_ver = check_fedml_is_latest_version(in_version)
        should_upgrade = False if fedml_is_latest_version else True
    except Exception as e:
        return

    if not should_upgrade:
        return
    upgrade_version = remote_ver

    do_upgrade(in_version, upgrade_version, show_local_console=True)


def run_cmd(command, show_local_console=False):
    process = ClientConstants.exec_console_with_script(command, should_capture_stdout=True,
                                                       should_capture_stderr=True)
    ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(process)
    if ret_code is None or ret_code <= 0:
        if out is not None:
            try:
                out_str = decode_byte_str(out)
            except:
                logging.info("utf-8 could not decode the output msg")
                out_str = ""
            if out_str != "":
                logging.info("{}".format(out_str))
                if show_local_console:
                    print(out_str)

        log_return_info(command, 0)

        is_cmd_run_ok = True
    else:
        if err is not None:
            try:
                err_str = decode_byte_str(err)
            except:
                logging.info("utf-8 could not decode the err msg")
                err_str = ""
            if err_str != "":
                logging.error("{}".format(err_str))
                if show_local_console:
                    print(err_str)

        log_return_info(command, ret_code)

        is_cmd_run_ok = False

    return is_cmd_run_ok


def get_local_fedml_version(fedml_init_file):
    fedml_version = fedml.__version__
    with open(fedml_init_file, "r") as f:
        while True:
            init_line = f.readline()
            if init_line is None:
                break

            if init_line.find("__version__") != -1:
                line_splits = init_line.split('"')
                if len(line_splits) >= 3:
                    fedml_version = line_splits[1]
                    break

    return fedml_version


def do_upgrade(config_version, upgrade_version, show_local_console=False):
    python_ver_major = sys.version_info[0]
    python_ver_minor = sys.version_info[1]
    is_pyton_37 = False
    if python_ver_major == 3 and python_ver_minor == 7:
        is_pyton_37 = True

    run_cmd("pip uninstall -y fedml", show_local_console=show_local_console)
    run_cmd("pip3 uninstall -y fedml", show_local_console=show_local_console)
    resolver_option = "--use-deprecated=legacy-resolver"

    fedml_init_file = os.path.abspath(fedml.__file__)

    if config_version == "release":
        run_cmd("pip install fedml=={} {}".format(
            upgrade_version, resolver_option if is_pyton_37 else ""
        ), show_local_console=show_local_console)

        local_fedml_version = get_local_fedml_version(fedml_init_file)
        upgrade_result = True if local_fedml_version == upgrade_version else False
        if not upgrade_result:
            run_cmd("pip3 install fedml=={} {}".format(
                upgrade_version, resolver_option if is_pyton_37 else ""
            ), show_local_console=show_local_console)

            local_fedml_version = get_local_fedml_version(fedml_init_file)
            upgrade_result = True if local_fedml_version == upgrade_version else False
    else:
        run_cmd("pip install --index-url https://test.pypi.org/simple/ "
                "--extra-index-url https://pypi.org/simple fedml=={} {}".
                format(upgrade_version, resolver_option if is_pyton_37 else ""),
                show_local_console=show_local_console)

        local_fedml_version = get_local_fedml_version(fedml_init_file)
        upgrade_result = True if local_fedml_version == upgrade_version else False
        if not upgrade_result:
            run_cmd("pip3 install --index-url https://test.pypi.org/simple/ "
                    "--extra-index-url https://pypi.org/simple fedml=={} {}".
                    format(upgrade_version, resolver_option if is_pyton_37 else ""),
                    show_local_console=show_local_console)

            local_fedml_version = get_local_fedml_version(fedml_init_file)
            upgrade_result = True if local_fedml_version == upgrade_version else False

    if upgrade_result:
        logging.info("Upgrade successfully")
    else:
        logging.info("Upgrade error")

    return upgrade_result


def is_runner_finished_normally(process_id):
    log_runner_result_file = os.path.join(expanduser("~"), ".fedml", "fedml_trace", str(process_id))
    if os.path.exists(log_runner_result_file):
        os.remove(log_runner_result_file)
        return True

    return False


def run_subprocess_open(shell_script_list):
    if platform.system() == 'Windows':
        script_process = subprocess.Popen(shell_script_list, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        script_process = subprocess.Popen(shell_script_list, preexec_fn=os.setsid)

    return script_process


def decode_our_err_result(out_err):
    try:
        result_str = decode_byte_str(out_err)
        return result_str
    except Exception as e:
        return out_err


def get_sys_realtime_stats():
    sys_mem = psutil.virtual_memory()
    total_mem = sys_mem.total
    free_mem = sys_mem.available
    total_disk_size = psutil.disk_usage("/").total
    free_disk_size = psutil.disk_usage("/").free
    cup_utilization = psutil.cpu_percent()
    cpu_cores = psutil.cpu_count()
    gpu_cores_total, _ = get_gpu_count_vendor()
    gpu_cores_total = len(get_gpu_list())

    gpu_available_ids = get_available_gpu_id_list(limit=gpu_cores_total)
    gpu_cores_available = len(gpu_available_ids) if gpu_available_ids is not None else 0
    net = psutil.net_io_counters()
    sent_bytes = net.bytes_sent
    recv_bytes = net.bytes_recv
    return total_mem, free_mem, total_disk_size, free_disk_size, cup_utilization, cpu_cores, gpu_cores_total, \
        gpu_cores_available, sent_bytes, recv_bytes, gpu_available_ids


def decode_byte_str(bytes_str):
    if not isinstance(bytes_str, bytes):
        return bytes_str

    encoding = dict()
    try:
        encoding = chardet.detect(bytes_str)
    except Exception as e:
        encoding = dict()
    str_encoding = encoding.get("encoding", 'utf-8')
    str_encoding = "utf-8" if str_encoding is None else str_encoding
    decoded_str = bytes_str.decode(encoding=str_encoding, errors='ignore')
    return decoded_str


def random1(msg, in_msg):
    msg_bytes = msg.encode('utf-8')
    in_msg_bytes = in_msg.encode('utf-8')
    out_bytes = bytearray()
    for i in range(len(msg_bytes)):
        out_bytes.append(msg_bytes[i] ^ in_msg_bytes[i % len(in_msg_bytes)])
    return out_bytes.hex()


def random2(msg, in_msg):
    msg_bytes = bytes.fromhex(msg)
    in_bytes = in_msg.encode('utf-8')
    out_bytes = bytearray()
    for i in range(len(msg_bytes)):
        out_bytes.append(msg_bytes[i] ^ in_bytes[i % len(in_bytes)])
    return out_bytes.decode('utf-8')


def get_file_encoding(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()
        f.close()

    try:
        encoding = chardet.detect(file_content)
    except Exception as e:
        encoding = dict()
    str_encoding = encoding.get("encoding", 'utf-8')
    str_encoding = "utf-8" if str_encoding is None else str_encoding

    return str_encoding


def read_gitignore_file(gitignore_file, ):
    try:
        ignore_list = list()
        with open(gitignore_file, "r") as ignore_file_handle:
            while True:
                ignore_line = ignore_file_handle.readline()
                if not ignore_line:
                    break
                ignore_line = ignore_line.replace('\n', '')
                if ignore_line.startswith("#") or len(ignore_line.lstrip(' ').rstrip(' ')) == 0:
                    continue
                ignore_list.append(ignore_line)

            ignore_file_handle.close()
            if len(ignore_list) > 0:
                ignore_list_str = ','.join(ignore_list)
                ignore_list_str = ignore_list_str.replace("\n", "")
                return ignore_list_str
    except Exception as e:
        pass

    return None


def get_host_ip():
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 53))
        ip = s.getsockname()[0]
    finally:
        s.close()
        return ip


def check_port(host, port):
    import socket

    s = socket.socket()
    try:
        s.connect((host, port))
        return True
    except:
        return False
    finally:
        s.close()


def get_available_port():
    for port in range(40000, 65535):
        if not check_port("localhost", port):
            return port

    return 40000


if __name__ == '__main__':
    fedml_is_latest_version, local_ver, remote_ver = check_fedml_is_latest_version("dev")
    print("FedML is latest version: {}, local version {}, remote version {}".format(
        fedml_is_latest_version, local_ver, remote_ver))
    # do_upgrade("release", remote_ver)
