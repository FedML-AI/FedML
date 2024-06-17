
from ..master.server_constants import ServerConstants
from .server_data_interface import FedMLServerDataInterface
from .master_protocol_manager import FedMLLaunchMasterProtocolManager
from .base_master_agent import FedMLBaseMasterAgent


class FedMLLaunchMasterAgent(FedMLBaseMasterAgent):

    def __init__(self):
        FedMLBaseMasterAgent.__init__(self)

    # Override
    def _get_log_file_dir(self):
        return ServerConstants.get_log_file_dir()

    # Override
    def _save_agent_info(self, unique_device_id, edge_id):
        ServerConstants.save_runner_infos(unique_device_id, edge_id)

    # Override
    def _init_database(self):
        FedMLServerDataInterface.get_instance().create_job_table()

    # Override
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return FedMLLaunchMasterProtocolManager(args, agent_config=agent_config)

