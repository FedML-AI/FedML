
from fedml.computing.scheduler.scheduler_core.account_manager import FedMLAccountManager
from fedml.computing.scheduler.slave.slave_agent import FedMLLaunchSlaveAgent
from fedml.computing.scheduler.master.master_agent import FedMLLaunchMasterAgent
from fedml.computing.scheduler.model_scheduler.model_device_server import FedMLDeployMasterAgent
from fedml.computing.scheduler.model_scheduler.model_device_client import FedMLDeployWorkerAgent
from fedml.core.common.singleton import Singleton


class FedMLUnitedAgent(Singleton):

    @staticmethod
    def get_instance():
        return FedMLUnitedAgent()

    def logout(self):
        FedMLLaunchSlaveAgent.logout()

    def login(self, userid, api_key=None, device_id=None,
              os_name=None, need_to_check_gpu=False, role=None, runner_cmd=None):
        # Create the launch master/slave and deploy master/slave agents.
        launch_slave_agent = FedMLLaunchSlaveAgent()
        launch_master_agent = FedMLLaunchMasterAgent()
        deploy_slave_agent = FedMLDeployWorkerAgent()
        deploy_master_agent = FedMLDeployMasterAgent()

        # Login with the launch slave role
        launch_slave_agent.login(
            api_key, api_key=api_key, device_id=device_id,
            os_name=os_name, role=role
        )

        # Get the communication manager, sender message queue and status center queue
        shared_communication_mgr = launch_slave_agent.get_protocol_manager().get_get_protocol_communication_manager()
        shared_sender_message_queue = launch_slave_agent.get_protocol_manager().get_protocol_sender_message_queue()
        shared_status_center_queue = launch_slave_agent.get_protocol_manager().get_get_protocol_status_center_queue()

        # Login with the launch master role based on the shared communication manager
        launch_master_agent.login(
            api_key, api_key=api_key, device_id=device_id,
            os_name=os_name, runner_cmd=runner_cmd,
            role=FedMLAccountManager.ROLE_GPU_MASTER_SERVER,
            communication_manager=shared_communication_mgr,
            sender_message_queue=shared_sender_message_queue,
            status_center_queue=shared_status_center_queue
        )

        # Login with the deployment master role based on the shared communication manager
        deploy_master_agent.login(
            userid, api_key=api_key, device_id=device_id,
            os_name=os_name, role=FedMLAccountManager.ROLE_DEPLOY_MASTER_ON_PREM,
            communication_manager=shared_communication_mgr,
            sender_message_queue=shared_sender_message_queue,
            status_center_queue=shared_status_center_queue
        )

        # Login with the deployment slave role based on the shared communication manager
        deploy_slave_agent.login(
            userid, api_key=api_key, device_id=device_id,
            os_name=os_name, role=FedMLAccountManager.ROLE_DEPLOY_WORKER_ON_PREM,
            communication_manager=shared_communication_mgr,
            sender_message_queue=shared_sender_message_queue,
            status_center_queue=shared_status_center_queue
        )

        # Start the slave agent to connect to servers and loop forever.
        launch_slave_agent.start()
