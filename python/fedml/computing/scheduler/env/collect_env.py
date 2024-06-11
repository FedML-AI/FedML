import os
import traceback

import fedml
import dotenv
from fedml.computing.scheduler.comm_utils.hardware_utils import HardwareUtil
from fedml.computing.scheduler.slave.client_diagnosis import ClientDiagnosis
from ..slave.client_constants import ClientConstants


def collect_env():
    print("\n======== FedML (https://fedml.ai) ========")
    print("FedML version: " + str(fedml.__version__))
    env_version = fedml.get_env_version()
    print("FedML ENV version: " + str(env_version))

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
        mpi_obj = mpi4py.MPI
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
        gpus = HardwareUtil.get_gpus()
        memory_total = 0.0
        memory_free = 0.0
        gpu_name = ""
        vendor = ""
        for gpu in gpus:
            memory_total += gpu.memoryTotal
            memory_free += gpu.memoryFree
            gpu_name = gpu.name
            vendor = gpu.vendor

        print(f"{vendor} GPU Info: " + gpu_name)
        print("Available GPU memory: {:.1f} G / {:.1f}G".format(
            memory_free / 1024.0, memory_total / 1024.0))

        device_count = len(gpus)
        print("device_count = {}".format(device_count))

        import torch

        torch_is_available = torch.cuda.is_available()
        print("torch_is_available = {}".format(torch_is_available))

    except:
        print("No GPU devices")

    try:
        print("\n======== Network Connection Checking ========")
        is_open_connected = ClientDiagnosis.check_open_connection(None)
        url = fedml._get_backend_service()
        if is_open_connected:
            print(f"The connection to {url} is OK.\n")
        else:
            print(f"You can not connect to {url}.\n")

        is_s3_connected = ClientDiagnosis.check_s3_connection(None)
        if is_s3_connected:
            print("The connection to S3 Object Storage is OK.\n")
        else:
            print("You can not connect to S3 Object Storage.\n")

        is_mqtt_connected = ClientDiagnosis.check_mqtt_connection()
        mqtt_url = fedml._get_mqtt_service()
        if is_mqtt_connected:
            print(f"The connection to {mqtt_url} (port:1883) is OK.\n")
        else:
            print(f"You can not connect to {mqtt_url}.\n")
    except Exception as e:
        print(f"The connection exception: {traceback.format_exc()}")
        pass


def get_env_file():
    global_services_dir = ClientConstants.get_global_services_dir()
    env_config_file = os.path.join(global_services_dir, ".env")
    # Create file if not exists
    if not os.path.exists(env_config_file):
        with open(env_config_file, 'w') as f:
            f.write("")
    return env_config_file


def load_env():
    env_config_file = get_env_file()
    dotenv.load_dotenv(dotenv_path=env_config_file, override=True)


def set_env_kv(key, value):
    os.environ[key] = value
    env_config_file = get_env_file()
    dotenv.set_key(env_config_file, key, value)
    load_env()
