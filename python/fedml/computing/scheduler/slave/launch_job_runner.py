from abc import ABC

from .base_slave_job_runner import FedMLBaseSlaveJobRunner
from .client_constants import ClientConstants


class FedMLLaunchSlaveJobRunner(FedMLBaseSlaveJobRunner, ABC):

    def __init__(self, args, edge_id=0, request_json=None, agent_config=None, run_id=0,
                 cuda_visible_gpu_ids_str=None):
        FedMLBaseSlaveJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=ClientConstants.get_data_dir(),
            agent_package_download_dir=ClientConstants.get_package_download_dir(),
            agent_package_unzip_dir=ClientConstants.get_package_unzip_dir(),
            agent_log_file_dir=ClientConstants.get_log_file_dir()
        )

    # Override
    def _generate_job_runner_instance(self, args, run_id=None, request_json=None, agent_config=None, edge_id=None):
        return FedMLLaunchSlaveJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=self.agent_config, edge_id=edge_id
        )

    # Override
    def _generate_extend_queue_list(self):
        return None

    # Override
    def get_download_package_info(self, packages_config=None):
        return super().get_download_package_info(packages_config)

    # Override
    def run_impl(
            self, run_extend_queue_list, sender_message_center,
            listener_message_queue, status_center_queue
    ):
        super().run_impl(
            run_extend_queue_list, sender_message_center,
            listener_message_queue, status_center_queue)

