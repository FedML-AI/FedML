import os


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
