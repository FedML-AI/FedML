
from .base_slave_agent import FedMLBaseSlaveAgent
from .client_constants import ClientConstants
from .client_data_interface import FedMLClientDataInterface
from .slave_protocol_manager import FedMLLaunchSlaveProtocolManager


class FedMLLaunchSlaveAgent(FedMLBaseSlaveAgent):
    def __init__(self):
        FedMLBaseSlaveAgent.__init__(self)

    # Override
    def _get_log_file_dir(self):
        return ClientConstants.get_log_file_dir()

    # Override
    def _save_agent_info(self, unique_device_id, edge_id):
        ClientConstants.save_runner_infos(unique_device_id, edge_id)

    # Override
    def _init_database(self):
        FedMLClientDataInterface.get_instance().create_job_table()

    # Override
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return FedMLLaunchSlaveProtocolManager(args, agent_config=agent_config)
