import os
import signal
from os.path import expanduser

import click
import psutil
import yaml
from .yaml_utils import load_yaml_config


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
        cpu_usage = "{:.0f}%".format((load15 / os.cpu_count()) * 100, 4)
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


def generate_yaml_doc(yaml_object, yaml_file):
    try:
        file = open(yaml_file, "w", encoding="utf-8")
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
    python_program = "python"
    python_version_str = os.popen("python --version").read()
    if python_version_str.find("Python 3.") == -1:
        python_version_str = os.popen("python3 --version").read()
        if python_version_str.find("Python 3.") != -1:
            python_program = "python3"

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


def cleanup_all_fedml_processes(login_program, exclude_login=False):
    # Cleanup all fedml relative processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=["pid", "name", "cmdline"])
            for cmd in pinfo["cmdline"]:
                if exclude_login:
                    if str(cmd).find("fedml_config.yaml") != -1:
                        click.echo("find fedml process at {}.".format(process.pid))
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        # process.terminate()
                        # process.join()
                else:
                    if (
                            str(cmd).find(login_program) != -1
                            or str(cmd).find("fedml_config.yaml") != -1
                    ):
                        click.echo("find fedml process at {}.".format(process.pid))
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        # process.terminate()
                        # process.join()
        except Exception as e:
            pass