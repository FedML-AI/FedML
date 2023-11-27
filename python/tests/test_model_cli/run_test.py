import argparse
import json
import os
import uuid

import fedml
from fedml.api.modules import model
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from fedml.computing.scheduler.model_scheduler import device_client_constants


def test_model_create_push(config_version="release"):
    cur_dir = os.path.dirname(__file__)
    model_config = os.path.join(cur_dir, "llm_deploy", "serving.yaml")
    model_name = f"test_model_{str(uuid.uuid4())}"
    fedml.set_env_version(config_version)
    model.create(model_name, model_config=model_config)
    model.push(
        model_name, api_key="10e87dd6d6574311a80200455e4d9b30",
        tag_list=[{"tagId": 147, "parentId": 3, "tagName": "LLM"}])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", "-v", type=str, default="dev")
    parser.add_argument("--log_file_dir", "-l", type=str, default="~")
    args = parser.parse_args()

    print("Hi everyone, I am testing the model cli.\n")

    fedml.set_env_version("dev")

    # endpoint_id = 1682
    # fedml.set_env_version("dev")
    # fedml.mlops.log_endpoint_status(
    #     endpoint_id, device_client_constants.ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)

    fedml.mlops.log_run_log_lines(1685, 0, ["failed to upload logs4"],
                                  log_source="MODEL_END_POINT")

    # args.log_file_dir = ClientConstants.get_log_file_dir()
    # args.run_id = 0
    # args.role = "client"
    # client_ids = list()
    # client_ids.append(111)
    # args.client_id_list = json.dumps(client_ids)
    # setattr(args, "using_mlops", True)
    # MLOpsRuntimeLog.get_instance(args).init_logs(show_stdout_log=False)
    # print("log 1")
    # MLOpsRuntimeLog.get_instance(args).enable_show_log_to_stdout()
    # print("log 2")
    # MLOpsRuntimeLog.get_instance(args).enable_show_log_to_stdout(enable=False)
    # print("log 3")

    # sys_utils.cleanup_model_monitor_processes(1627, "ep-1124-304-13ad33",
    #                                           "", "", "")
    #
    # test_model_create_push()

