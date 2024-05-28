
import json
from fedml.core.common.singleton import Singleton
from ..master.base_master_job_runner_manager import FedMLBaseMasterJobRunnerManager
from .master_job_runner import FedMLDeployMasterJobRunner
from ..scheduler_core.general_constants import GeneralConstants


class FedMLDeployJobRunnerManager(FedMLBaseMasterJobRunnerManager, Singleton):
    def __init__(self):
        FedMLBaseMasterJobRunnerManager.__init__(self)

    @staticmethod
    def get_instance():
        return FedMLDeployJobRunnerManager()

    # Override
    def _generate_job_runner_instance(
            self, args, run_id=None, request_json=None, agent_config=None, edge_id=None
    ):
        job_runner = FedMLDeployMasterJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=agent_config, edge_id=edge_id)
        job_runner.infer_host = GeneralConstants.get_ip_address(request_json)
        return job_runner

    def save_deployment_result(self, topic, payload):
        payload_json = json.loads(payload)
        endpoint_id = payload_json["end_point_id"]
        run_id_str = str(endpoint_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].save_deployment_result(topic=topic, payload=payload)

    def send_deployment_stages(
            self, end_point_id, model_name, model_id, model_inference_url,
            model_stages_index, model_stages_title, model_stage_detail, message_center=None
    ):
        run_id_str = str(end_point_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].send_deployment_stages(
                end_point_id, model_name, model_id, model_inference_url,
                model_stages_index, model_stages_title, model_stage_detail,
                message_center=message_center
            )

    def send_deployment_delete_request_to_edges(self, end_point_id, payload, model_msg_object, message_center=None):
        run_id_str = str(end_point_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].send_deployment_delete_request_to_edges(
                payload, model_msg_object, message_center=message_center)

    def stop_device_inference_monitor(self, run_id, end_point_name, model_id, model_name, model_version):
        run_id_str = str(run_id)
        if self.job_runners.get(run_id_str, None) is not None:
            self.job_runners[run_id_str].stop_device_inference_monitor(
                run_id, end_point_name, model_id, model_name, model_version)

    @staticmethod
    def recover_inference_and_monitor():
        FedMLDeployMasterJobRunner.recover_inference_and_monitor()

    @staticmethod
    def generate_request_json_with_replica_num_diff(run_id, edge_id, request_json):
        return FedMLDeployMasterJobRunner.generate_request_json_with_replica_num_diff(run_id, edge_id, request_json)

    @staticmethod
    def generate_request_json_with_replica_version_diff(run_id, edge_id, request_json):
        return FedMLDeployMasterJobRunner.generate_request_json_with_replica_version_diff(run_id, edge_id, request_json)
