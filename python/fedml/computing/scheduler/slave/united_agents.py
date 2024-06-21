from fedml.computing.scheduler.model_scheduler.master_agent import FedMLDeployMasterAgent
from fedml.computing.scheduler.model_scheduler.worker_agent import FedMLDeployWorkerAgent
from fedml.computing.scheduler.scheduler_core.account_manager import FedMLAccountManager
from fedml.computing.scheduler.slave.slave_agent import FedMLLaunchSlaveAgent
from fedml.computing.scheduler.master.master_agent import FedMLLaunchMasterAgent
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
        login_result = launch_slave_agent.login(
            api_key, api_key=api_key, device_id=device_id,
            os_name=os_name, role=role
        )

        # Get the communication manager, sender message queue
        shared_communication_mgr = launch_slave_agent.get_protocol_manager().get_protocol_communication_manager()
        shared_slave_sender_message_queue = launch_slave_agent.get_protocol_manager().get_protocol_sender_message_queue()
        shared_slave_sender_message_event = launch_slave_agent.get_protocol_manager().get_protocol_sender_message_event()

        # Login with the launch master role based on
        # the shared communication manager, sender message center
        launch_master_agent.login(
            api_key, api_key=api_key, device_id=login_result.device_id,
            os_name=os_name, runner_cmd=runner_cmd,
            role=FedMLAccountManager.ROLE_GPU_MASTER_SERVER,
            communication_manager=shared_communication_mgr,
            sender_message_queue=None
        )

        # Get the status center queue
        shared_slave_status_center_queue = launch_slave_agent.get_protocol_manager().get_protocol_status_center_queue()
        shared_master_status_center_queue = launch_master_agent.get_protocol_manager().get_protocol_status_center_queue()
        shared_master_sender_message_queue = launch_master_agent.get_protocol_manager().get_protocol_sender_message_queue()
        shared_master_sender_message_event = launch_master_agent.get_protocol_manager().get_protocol_sender_message_event()

        # Login with the deployment master role based on
        # the shared communication manager, sender message center, status center
        deploy_master_login_result = deploy_master_agent.login(
            userid, api_key=api_key, device_id=login_result.device_id,
            os_name=os_name, role=FedMLAccountManager.ROLE_DEPLOY_MASTER_ON_PREM,
            communication_manager=shared_communication_mgr
        )

        # Login with the deployment slave role based on
        # the shared communication manager, sender message center, status center
        deploy_slave_login_result = deploy_slave_agent.login(
            userid, api_key=api_key, device_id=login_result.device_id,
            os_name=os_name, role=FedMLAccountManager.ROLE_DEPLOY_WORKER_ON_PREM,
            communication_manager=shared_communication_mgr
        )

        # Set the deployment ids to launch agent so that we can report the related device info to MLOps.
        launch_slave_agent.save_deploy_ids(
            deploy_master_edge_id=deploy_master_login_result.edge_id,
            deploy_slave_edge_id=deploy_slave_login_result.edge_id)

        # Start the slave agent to connect to servers and loop forever.
        launch_slave_agent.start()
