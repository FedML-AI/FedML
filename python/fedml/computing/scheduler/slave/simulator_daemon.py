import argparse
import time

from fedml import mlops, FEDML_TRAINING_PLATFORM_SIMULATION
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.master.server_constants import ServerConstants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="client")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--rank", "-rk", type=str, default="1")
    parser.add_argument("--enable_wandb", "-ew", type=bool, default=False)
    parser.add_argument("--using_mlops", "-um", type=bool, default=True)
    parser.add_argument("--log_file_dir", "-lfd", type=str, default="")
    parser.add_argument("--config_version", "-cf", type=str, default="release")
    parser.add_argument("--client_id", "-ci", type=str, default="")
    args = parser.parse_args()

    setattr(args, "enable_tracking", True)
    setattr(args, "training_type", FEDML_TRAINING_PLATFORM_SIMULATION)
    setattr(args, "simulator_daemon", True)
    mlops.init(args)

    while True:
        simulator_processes, status_info = sys_utils.get_simulator_process_list(ClientConstants.get_data_dir(),
                                                                                ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        for process_id, run_id in simulator_processes.items():
            run_status = status_info.get(str(run_id), "")
            if not sys_utils.simulator_process_is_running(process_id):
                if run_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
                    sys_utils.remove_simulator_process(ClientConstants.get_data_dir(),
                                                       ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                                       process_id)
                else:
                    mlops.log_training_failed_status(run_id)
                    time.sleep(0.1)
                    mlops.log_training_failed_status(run_id)
                    time.sleep(0.1)
                    mlops.log_aggregation_failed_status(run_id)
                    time.sleep(0.1)
                    mlops.log_aggregation_failed_status(run_id)
                    time.sleep(0.1)
                    mlops.log_aggregation_failed_status(run_id)
                    sys_utils.remove_simulator_process(ClientConstants.get_data_dir(),
                                                       ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                                       process_id)

        time.sleep(1)
