import os
import signal
import traceback
from os.path import expanduser

import click
import psutil
import yaml
from .yaml_utils import load_yaml_config


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
        import nvidia_smi

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_info = str(handle)
        gpu_available_mem = "{:.1f} G".format(info.free / 1024 / 1024 / 1024)
        gpu_total_mem = "{:.1f}G".format(info.total / 1024 / 1024 / 1024)
        nvidia_smi.nvmlShutdown()
    except:
        pass

    return fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
           cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem


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


def get_running_info(cs_home_dir, cs_info_dir):
    home_dir = expanduser("~")
    runner_info_file = os.path.join(
        home_dir, cs_home_dir, "fedml", "data", cs_info_dir, "runner_infos.yaml"
    )
    if os.path.exists(runner_info_file):
        running_info = load_yaml_config(runner_info_file)
        return running_info["run_id"], running_info["edge_id"]
    return 0, 0


def get_python_program():
    python_program = "python3"
    python_version_str = os.popen("python3 --version").read()
    if python_version_str.find("Python 3.") == -1:
        python_program = "python"

    return python_program


def cleanup_login_process(runner_home_dir, runner_info_dir):
    try:
        home_dir = expanduser("~")
        local_pkg_data_dir = os.path.join(home_dir, runner_home_dir, "fedml", "data")
        edge_process_id_file = os.path.join(
            local_pkg_data_dir, runner_info_dir, "runner-process.id"
        )
        edge_process_info = load_yaml_config(edge_process_id_file)
        edge_process_id = edge_process_info.get("process_id", None)
        if edge_process_id is not None:
            edge_process = psutil.Process(edge_process_id)
            if edge_process is not None:
                os.killpg(os.getpgid(edge_process.pid), signal.SIGTERM)
                # edge_process.terminate()
                # edge_process.join()
        yaml_object = {}
        yaml_object["process_id"] = -1
        generate_yaml_doc(yaml_object, edge_process_id_file)

    except Exception as e:
        pass


def save_login_process(runner_home_dir, runner_info_dir, edge_process_id):
    home_dir = expanduser("~")
    local_pkg_data_dir = os.path.join(home_dir, runner_home_dir, "fedml", "data")
    try:
        os.makedirs(local_pkg_data_dir)
    except Exception as e:
        pass
    try:
        os.makedirs(os.path.join(local_pkg_data_dir, runner_info_dir))
    except Exception as e:
        pass

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
                click.echo("find client learning process at {}.".format(process.pid))
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
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
                click.echo("find client diagnosis process at {}.".format(process.pid))
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            pass


def cleanup_all_fedml_client_login_processes(login_program):
    # Cleanup all fedml client login processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            for cmd in pinfo["cmdline"]:
                if str(cmd).find(login_program) != -1:
                    if os.path.basename(cmd) == login_program:
                        click.echo("find client login process at {}.".format(process.pid))
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
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
                click.echo("find server learning process at {}.".format(process.pid))
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            pass


def cleanup_all_fedml_client_api_processes():
    # Cleanup all fedml client api processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            find_api_process = False
            for cmd in pinfo["cmdline"]:
                if str(cmd).find("client_api:api") != -1:
                    find_api_process = True

            if find_api_process:
                click.echo("find client api process at {}.".format(process.pid))
                os.kill(process.pid, signal.SIGTERM)
        except Exception as e:
            pass


def cleanup_all_fedml_server_api_processes():
    # Cleanup all fedml server api processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            find_api_process = False
            for cmd in pinfo["cmdline"]:
                if str(cmd).find("server_api:api") != -1:
                    find_api_process = True

            if find_api_process:
                click.echo("find server api process at {}.".format(process.pid))
                os.kill(process.pid, signal.SIGTERM)
        except Exception as e:
            pass


def cleanup_all_fedml_server_login_processes(login_program):
    # Cleanup all fedml client login processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            for cmd in pinfo["cmdline"]:
                if str(cmd).find(login_program) != -1:
                    if os.path.basename(cmd) == login_program:
                        click.echo("find server login process at {}.".format(process.pid))
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            pass


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
    try:
        os.makedirs(simulator_proc_path)
    except Exception as e:
        pass

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


def remove_simulator_process(data_dir, runner_info_dir, process_id):
    simulator_proc_path = os.path.join(data_dir, runner_info_dir, "simulator-processes")
    try:
        os.makedirs(simulator_proc_path)
    except Exception as e:
        pass

    try:
        simulator_process_id_file = os.path.join(
            simulator_proc_path, "simulator-process-{}".format(str(process_id))
        )
        os.remove(simulator_process_id_file)
    except Exception as e:
        pass


def simulator_process_is_running(process_id):
    for process in psutil.process_iter():
        if str(process.pid) == str(process_id):
            return True

    return False


def log_return_info(bootstrap_file, ret_code):
    import logging
    err_desc = SYS_ERR_CODE_MAP.get(str(ret_code), "")
    if ret_code == 0:
        logging.info("Run {} return code {}. {}".format(
            bootstrap_file, ret_code, err_desc))
    else:
        fatal_err_desc = SYS_ERR_CODE_MAP.get(str(ret_code), "")
        if ret_code >= FETAL_ERROR_START_CODE and fatal_err_desc == "":
            fatal_err_desc = SYS_ERR_CODE_MAP.get(str(FETAL_ERROR_START_CODE))

        logging.error("Run {} return code {}. {}".format(
            bootstrap_file, ret_code, fatal_err_desc if fatal_err_desc != "" else err_desc))

