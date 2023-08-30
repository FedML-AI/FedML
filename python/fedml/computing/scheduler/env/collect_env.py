import os

import fedml
from fedml.computing.scheduler.slave.client_diagnosis import ClientDiagnosis


def collect_env(args=None):
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

        import torch

        torch_is_available = torch.cuda.is_available()
        print("torch_is_available = {}".format(torch_is_available))

        device_count = torch.cuda.device_count()
        print("device_count = {}".format(device_count))

        device_name = torch.cuda.get_device_name(0)
        print("device_name = {}".format(device_name))

    except:
        print("No GPU devices")

    print("\n======== Network Connection Checking ========")
    is_open_connected = ClientDiagnosis.check_open_connection(args)
    if is_open_connected:
        print("The connection to https://open{}.fedml.ai is OK.".format("-dev" if args is not None and hasattr(args, "config_version") and args.config_version == "dev" else ""))
    else:
        print("You can not connect to https://open{}.fedml.ai.".format("-dev" if args is not None and hasattr(args, "config_version") and args.config_version == "dev" else ""))

    is_s3_connected = ClientDiagnosis.check_s3_connection(args)
    if is_s3_connected:
        print("The connection to AWS S3 is OK.")
    else:
        print("You can not connect to AWS S3.")

    is_mqtt_connected = ClientDiagnosis.check_mqtt_connection(args)
    if is_mqtt_connected:
        print("The connection to mqtt{}.fedml.ai (port:1883) is OK.".format("-dev" if args is not None and hasattr(args, "config_version") and args.config_version == "dev" else ""))
    else:
        print("You can not connect to mqtt{}.fedml.ai (port:1883).".format("-dev" if args is not None and hasattr(args, "config_version") and args.config_version == "dev" else ""))
