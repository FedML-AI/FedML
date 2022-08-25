import os

import fedml


def collect_env():
    print("\n======== FedML (https://fedml.ai) ========")
    print("FedML version: " + str(fedml.__version__))
    print("Execution path:" + str(os.path.abspath(fedml.__file__)))

    print("\n======== Running Environment ========")
    import platform

    print("OS: " + platform.platform())
    print("Hardware: " + platform.machine())

    import sys

    print("Python version: " + sys.version)

    try:
        import torch
        print("PyTorch version: " + torch.__version__)
    except:
        print("PyTorch is not installed properly")

    try:
        from mpi4py import MPI

        print("MPI4py is installed")
    except:
        print("MPI4py is NOT installed")

    print("\n======== CPU Configuration ========")

    try:
        import psutil

        # Getting loadover15 minutes
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = (load15 / os.cpu_count()) * 100

        print("The CPU usage is : {:.0f}%".format(cpu_usage))
        print(
            "Available CPU Memory: {:.1f} G / {}G".format(
                psutil.virtual_memory().available / 1024 / 1024 / 1024,
                psutil.virtual_memory().total / 1024 / 1024 / 1024,
            )
        )
    except:
        print("\n")

    try:
        print("\n======== GPU Configuration ========")
        import nvidia_smi

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("NVIDIA GPU Info: " + str(handle))
        print(
            "Available GPU memory: {:.1f} G / {}G".format(
                info.free / 1024 / 1024 / 1024, info.total / 1024 / 1024 / 1024
            )
        )
        nvidia_smi.nvmlShutdown()
    except:
        print("No GPU devices")
