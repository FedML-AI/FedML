import json
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.model_scheduler import device_client_constants
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.scheduler_entry.constants import Constants


class FedMLDeployJobLauncher:
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    STATUS_IDLE = "IDLE"

    def __init__(self, edge_id=None):
        self.edge_id = edge_id
        self.run_model_device_ids = dict()

    @staticmethod
    def deploy_model(serving_devices, request_json, run_id):
        run_config = request_json["run_config"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_type = job_yaml.get("job_type", None)
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        if job_type == Constants.JOB_TASK_TYPE_DEPLOY or job_type == Constants.JOB_TASK_TYPE_SERVE:
            # computing = job_yaml.get("computing", {})
            # num_gpus = computing.get("minimum_num_gpus", 1)
            serving_args = run_params.get("serving_args", {})
            model_id = serving_args.get("model_id", None)
            model_name = serving_args.get("model_name", None)
            model_version = serving_args.get("model_version", None)
            # model_storage_url = serving_args.get("model_storage_url", None)
            endpoint_name = serving_args.get("endpoint_name", None)
            endpoint_id = serving_args.get("endpoint_id", None)
            random = serving_args.get("random", "")
            random_out = sys_utils.random2(random, "FEDML@9999GREAT")
            random_list = random_out.split("FEDML@")
            device_type = device_client_constants.ClientConstants.login_role_list[
                device_client_constants.ClientConstants.LOGIN_MODE_FEDML_CLOUD_INDEX]
            FedMLModelCards.get_instance().deploy_model(
                model_name, device_type, json.dumps(serving_devices),
                "", random_list[1], None,
                in_model_id=model_id, in_model_version=model_version,
                endpoint_name=endpoint_name, endpoint_id=endpoint_id, run_id=run_id)
            return endpoint_id
        return None

    def check_model_device_ready_and_deploy(self, request_json, run_id, master_device_id,
                                            slave_device_id, run_edge_ids=None):
        run_config = request_json["run_config"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_type = job_yaml.get("job_type", None)
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        if job_type != Constants.JOB_TASK_TYPE_DEPLOY and job_type != Constants.JOB_TASK_TYPE_SERVE:
            return

        # Init model device ids for each run
        run_id_str = str(run_id)
        if self.run_model_device_ids.get(run_id_str, None) is None:
            self.run_model_device_ids[run_id_str] = list()

        # Append master device and slave devices to the model devices map
        self.run_model_device_ids[run_id_str].append({"master_device_id": master_device_id,
                                                      "slave_device_id": slave_device_id})
        model_device_ids = self.run_model_device_ids.get(run_id_str, None)
        if model_device_ids is None:
            return
        if run_edge_ids is None:
            return

        # Check if all model devices are ready
        if len(model_device_ids) != len(run_edge_ids.get(run_id_str, list())):
            return

        # Generate model master ids and model slave device ids
        device_master_ids = list()
        device_slave_ids = list()
        for device_ids in model_device_ids:
            model_master_id = device_ids.get("master_device_id")
            model_slave_id = device_ids.get("slave_device_id")
            device_master_ids.append(model_master_id)
            device_slave_ids.append(model_slave_id)

        if len(device_master_ids) <= 0:
            return

        # Generate serving devices for deploying
        serving_devices = list()
        serving_devices.append(device_master_ids[0])
        serving_devices.extend(device_slave_ids)

        # Start to deploy the model
        FedMLDeployJobLauncher.deploy_model(serving_devices, request_json, run_id=run_id)



