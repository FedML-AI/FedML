
from ..master.server_constants import ServerConstants
from ..scheduler_core.general_constants import GeneralConstants
from .base_master_job_runner import FedMLBaseMasterJobRunner


class FedMLLaunchMasterJobRunner(FedMLBaseMasterJobRunner):

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0,
                 cuda_visible_gpu_ids_str=None):
        FedMLBaseMasterJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=ServerConstants.get_data_dir(),
            agent_package_download_dir=ServerConstants.get_package_download_dir(),
            agent_package_unzip_dir=GeneralConstants.get_package_unzip_dir(ServerConstants.get_package_download_dir()),
            agent_log_file_dir=ServerConstants.get_log_file_dir()
        )

    # Override
    def _generate_job_runner_instance(self, args, run_id=None, request_json=None, agent_config=None, edge_id=None,):
        return FedMLLaunchMasterJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=agent_config, edge_id=edge_id
        )

    # Override
    def _generate_extend_queue_list(self):
        return None

    # Override
    def get_download_package_info(self, packages_config=None):
        return super().get_download_package_info(packages_config)

    # Override
    def run_impl(
            self, edge_id_status_queue, edge_device_info_queue, run_metrics_queue,
            run_event_queue, run_artifacts_queue, run_logs_queue, edge_device_info_global_queue,
            run_extend_queue_list=None, sender_message_queue=None, listener_message_queue=None,
            status_center_queue=None
    ):
        super().run_impl(
            edge_id_status_queue, edge_device_info_queue, run_metrics_queue,
            run_event_queue, run_artifacts_queue, run_logs_queue, edge_device_info_global_queue,
            run_extend_queue_list=run_extend_queue_list, sender_message_queue=sender_message_queue,
            listener_message_queue=listener_message_queue, status_center_queue=status_center_queue)
