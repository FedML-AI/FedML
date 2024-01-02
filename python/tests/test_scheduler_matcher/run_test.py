import argparse
import time

import pynvml

from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.scheduler_core.scheduler_matcher import SchedulerMatcher
from fedml.computing.scheduler.scheduler_core.compute_gpu_db import ComputeGpuDatabase
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.model_scheduler import device_model_deployment


def test_match_multi_nodes_with_multi_gpus(in_args, run_id, node_num=1, gpu_num_per_node=1, request_gpu_num=1):
    request_json = {'threshold': '20', 'starttime': 1698291772383, 'edgestates': '{}', 'edgeids': [705], 'urls': '"[]"',
                    'id': 4887, 'state': 'STARTING', 'projectid': 289,
                    'run_config':
                        {'configName': 'hello_world_FedMLLaunchApp',
                         'packages_config': {'linuxClient': 'client-package.zip', 'androidClientVersion': 0,
                                             'androidClientUrl': '', 'androidClient': '',
                                             'linuxClientUrl': 'https://fedml.s3.amazonaws.com/hello_world_FedMLLaunchApp%4009c7594a-7d88-4983-bc73-a852ae70392f?AWSAccessKeyId=AKIAUAWARWF4SW36VYXP&Signature=w0mXFtHBgO72aBAKjnrEEBbrZNY%3D&Expires=1729827768'},
                         'data_config': {}, 'userId': 'fedml-alex',
                         'parameters':
                             {'job_yaml':
                                  {'job_type': 'train', 'workspace': 'hello_world',
                                   'computing': {'resource_type': 'A100-80G', 'minimum_num_gpus': 3,
                                                 'maximum_cost_per_hour': '$3000'},
                                   'model_args': {'input_dim': '784',
                                                  'model_cache_path': '/Users/alexliang/fedml_models',
                                                  'model_name': 'lr', 'output_dim': '10'},
                                   'job_subtype': 'generate_training',
                                   'bootstrap': '# pip install -r requirements.txt\necho "Bootstrap finished."\n',
                                   'job': 'echo "current job id: $FEDML_CURRENT_RUN_ID"\necho "current edge id: $FEDML_CURRENT_EDGE_ID"\necho "master node address : $FEDML_NODE_0_ADDR"\necho "master node port : $FEDML_NODE_0_PORT"\necho "num nodes : $FEDML_NUM_NODES"\necho "Hello, Here is the launch platform."\necho "Current directory is as follows."\npwd\npython3 hello_world.py\n#sleep 20\n#exit 1\n#echo "Current GPU information is as follows."\n#nvidia-smi # Print GPU information\n#gpustat\n#echo "Download the file from http://212.183.159.230/200MB.zip ..."\n#wget http://212.183.159.230/200MB.zip\n#rm ./200MB.zip*\n#echo "The downloading task has finished."\n# echo "Training the vision transformer model using PyTorch..."\n# python vision_transformer.py --epochs 1\n',
                                   'data_args': {'dataset_type': 'csv', 'dataset_path': './dataset',
                                                 'dataset_name': 'mnist'}},
                              'environment_args': {'bootstrap': 'bootstrap.sh'}}},
                    'servers_state': '{}', 'timestamp': '1698291772419', 'cloud_agent_id': '706',
                    'lastupdatetime': 1698291772408, 'create_time': 1698291772408, 'groupid': 138,
                    'edges': [{'device_id': '0xT3630FW2YM@MacOS.Edge.Device', 'os_type': 'MacOS', 'id': 705}],
                    'server_id': '706', 'token': 'c9356b9c4ce44363bb66366b210201',
                    'name': 'b8dc1629-338e-495c-8943-76a1f1ee4868',
                    'creater': '214', 'runId': 4887, 'applicationId': 2070, 'group_server_id_list': [706],
                    'status': 0, 'is_retain': 0}
    edge_id_base = 700
    edge_id_list = list()
    active_edge_info_dict = dict()
    for index in range(node_num):
        edge_id = edge_id_base + index
        edge_id_list.append(edge_id)
        active_edge_info_dict[str(edge_id)] = {
            'edge_id': edge_id, 'memoryTotal': 64.0, 'memoryAvailable': 21.52,
            'diskSpaceTotal': 926.35, 'diskSpaceAvailable': 176.79, 'cpuUtilization': 74.0,
            'cpuCores': 10, 'gpuCoresTotal': gpu_num_per_node, 'gpuCoresAvailable': gpu_num_per_node,
            'gpu_available_ids': list(range(0, gpu_num_per_node)), 'node_ip': '192.168.68.102',
            'node_port': 40000, 'networkTraffic': 8444397568, 'updateTime': 1698291782320,
            'fedml_version': '0.8.8a156', 'user_id': '214', "gpu_list": sys_utils.get_gpu_list()}

    # Show request infos
    print(f"Node num {node_num}, gpu num per node {gpu_num_per_node}, request gpu num {request_gpu_num}")

    # Show current GPUs
    gpu_count, gpu_available_count = SchedulerMatcher.parse_and_print_gpu_info_for_all_edges(
        active_edge_info_dict, show_gpu_list=True)
    print("\n")

    a="1".split(',')
    gpu_list = JobRunnerUtils.trim_unavailable_gpu_ids(a)

    print(f"Occupy GPUs {request_gpu_num}.")
    gpu_ids = JobRunnerUtils.get_instance().occupy_gpu_ids(run_id, request_gpu_num)
    print(f"Run {run_id}, applied gpu ids {gpu_ids}, available GPU ids for: {JobRunnerUtils.get_instance().get_available_gpu_id_list()}")
    JobRunnerUtils.get_instance().release_gpu_ids(run_id)

    gpu_ids = JobRunnerUtils.get_instance().occupy_gpu_ids(103, 2)
    print(f"Run 103, applied gpu ids {gpu_ids}, available GPU ids: {JobRunnerUtils.get_instance().get_available_gpu_id_list()}")
    JobRunnerUtils.get_instance().release_gpu_ids(103)

    gpu_ids = JobRunnerUtils.get_instance().occupy_gpu_ids(104, 3)
    print(f"Run 104, applied gpu ids {gpu_ids}, available GPU ids: {JobRunnerUtils.get_instance().get_available_gpu_id_list()}")

    gpu_ids = JobRunnerUtils.get_instance().occupy_gpu_ids(105, 3)
    print(f"Run 105, applied gpu ids {gpu_ids}, available GPU ids: {JobRunnerUtils.get_instance().get_available_gpu_id_list()}")

    # Match and assign gpus to each device
    assigned_gpu_num_dict, assigned_gpu_ids_dict = SchedulerMatcher.match_and_assign_gpu_resources_to_devices(
        request_gpu_num, edge_id_list, active_edge_info_dict
    )
    if assigned_gpu_ids_dict is None or assigned_gpu_ids_dict is None:
        print(f"No resources available."
              f"Total available GPU count {gpu_available_count} is less than "
              f"request GPU count {request_gpu_num}")
        return

    # Generate new edge id list after matched
    edge_id_list = SchedulerMatcher.generate_new_edge_list_for_gpu_matching(assigned_gpu_num_dict)
    if len(edge_id_list) <= 0:
        print(f"Request parameter for GPU num is invalid."
              f"Total available GPU count {gpu_available_count}."
              f"Request GPU num {request_gpu_num}")
        return

    # Generate master node addr and port
    master_node_addr, master_node_port = SchedulerMatcher.get_master_node_info(edge_id_list, active_edge_info_dict)

    for edge_id in edge_id_list:
        scheduler_info = SchedulerMatcher.generate_match_info_for_scheduler(
            edge_id, edge_id_list, master_node_addr, master_node_port, assigned_gpu_num_dict, assigned_gpu_ids_dict
        )
        print(f"server: generate scheduler info for {edge_id}\n   {scheduler_info}")

        export_env_cmd_list, env_name_value_map = JobRunnerUtils.assign_matched_resources_to_run_and_generate_envs(
            run_id, "export", scheduler_info
        )
        print(f"client: assigned resources to run for edge id {edge_id}\n   {export_env_cmd_list}\n")

    print(f"Release GPUs {request_gpu_num}.")
    JobRunnerUtils.get_instance().release_gpu_ids(run_id)
    JobRunnerUtils.get_instance().release_gpu_ids(104)
    JobRunnerUtils.get_instance().release_gpu_ids(105)
    print(f"available GPU ids: {JobRunnerUtils.get_instance().get_available_gpu_id_list()}")


def test_config_map_to_env_variables():
    config_dict = sys_utils.load_yaml_config("./train_job.yaml")
    export_env_command_list, env_name_value_map = JobRunnerUtils.parse_config_args_as_env_variables(
        "export", config_dict)
    print(f"export env commands: {export_env_command_list}")
    print(f"environment name values: {env_name_value_map}")

    replaced_entry_commands = JobRunnerUtils.replace_entry_command_with_env_variable(
        ["${FEDML_DATA_ARGS_DATASET_NAME} is loading...",
         "$FEDML_DATA_ARGS_DATASET_NAME is loading...",
         "%FEDML_DATA_ARGS_DATASET_NAME% is loading..."], env_name_value_map)
    print(f"replaced entry commands: {replaced_entry_commands}")

    replaced_entry_args = JobRunnerUtils.replace_entry_args_with_env_variable(
        "python train.py --dataset_name ${FEDML_DATA_ARGS_DATASET_NAME}", env_name_value_map)
    print(f"replaced entry args: {replaced_entry_args}")


def test_gpu_info():
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    for i in range(0, gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            utilz = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            print(f"index {i}, utilz {utilz}, memory {memory}, temp {temp}.")
        except Exception as e:
            print(f"Exception when getting gpu info: {str(e)} ")


def test_gpu_db():
    ComputeGpuDatabase.get_instance().set_database_base_dir(ClientConstants.get_database_dir())
    ComputeGpuDatabase.get_instance().create_table()
    run_id = 2000
    device_id = 1111
    ComputeGpuDatabase.get_instance().set_device_run_gpu_ids(device_id, run_id, [0,1,2])
    gpu_ids = ComputeGpuDatabase.get_instance().get_device_run_gpu_ids(device_id, run_id)

    ComputeGpuDatabase.get_instance().set_device_run_num_gpus(device_id, run_id, 3)
    num_gpus = ComputeGpuDatabase.get_instance().get_device_run_num_gpus(device_id, run_id)

    ComputeGpuDatabase.get_instance().set_device_available_gpu_ids(device_id, [0,1,2])
    gpu_ids = ComputeGpuDatabase.get_instance().get_device_available_gpu_ids(device_id)

    ComputeGpuDatabase.get_instance().set_device_total_num_gpus(device_id, 3)
    total_num_gpus = ComputeGpuDatabase.get_instance().get_device_total_num_gpus(device_id)

    ComputeGpuDatabase.get_instance().set_run_device_ids(run_id, [0,1,2])
    gpu_ids = ComputeGpuDatabase.get_instance().get_run_device_ids(run_id)

    ComputeGpuDatabase.get_instance().set_run_total_num_gpus(run_id, 3)
    total_num_gpus = ComputeGpuDatabase.get_instance().get_run_total_num_gpus(run_id)

    ComputeGpuDatabase.get_instance().set_edge_model_id_map(run_id, 1122, 1133, 1144)
    edge_id, master_id, worker_id = ComputeGpuDatabase.get_instance().get_edge_model_id_map(run_id)
    print("OK")


def test_request_gpu_ids_on_deployment():
    gpu_ids, gpu_attach_cmd = device_model_deployment.request_gpu_ids_on_deployment(111, 2, 222)
    print(f"test_request_gpu_ids_on_deployment result: gpu_ids {gpu_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", "-v", type=str, default="dev")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="client")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")
    parser.add_argument("--no_gpu_check", "-ngc", type=int, default=1)
    args = parser.parse_args()
    args.current_running_dir = None
    run_id = 1000

    print("Hi everyone, I am testing the server runner.\n")

    test_request_gpu_ids_on_deployment()
    time.sleep(1000000)

    print("Test for mapping config dictionaries to environment variables.")
    test_config_map_to_env_variables()

    print("Test for single node with single GPU")
    test_match_multi_nodes_with_multi_gpus(args, run_id, node_num=1, gpu_num_per_node=8, request_gpu_num=1)

    # test_match_multi_nodes_with_multi_gpus(args, run_id, node_num=1, gpu_num_per_node=8, request_gpu_num=2)
    #
    # test_match_multi_nodes_with_multi_gpus(args, run_id, node_num=1, gpu_num_per_node=8, request_gpu_num=5)

    # print("Test for single node with multi GPUs")
    # test_match_multi_nodes_with_multi_gpus(args, run_id, node_num=1, gpu_num_per_node=3, request_gpu_num=1)
    #
    # print("Test for multi node with multi GPUs")
    # test_match_multi_nodes_with_multi_gpus(args, run_id, node_num=3, gpu_num_per_node=8, request_gpu_num=18)

