import os
import platform
import multiprocessing
import queue
import time
import logging
import traceback

from fedml.computing.scheduler.scheduler_core.message_common import FedMLMessageEntity, FedMLStatusEntity
from fedml.computing.scheduler.scheduler_core.status_center import FedMLStatusCenter, StatusCenterStoppedException
from fedml.computing.scheduler.scheduler_core.status_manager_protocols import FedMLStatusManager


class FedMLSlaveStatusCenter(FedMLStatusCenter):
    _status_queue = None

    @classmethod
    def initialize_status_queue(cls):
        if cls._status_queue is None:
            cls._status_queue = multiprocessing.Manager().Queue(-1)  # or any other appropriate initialization

    def __init__(self, message_center, is_deployment_status_center):
        if FedMLSlaveStatusCenter._status_queue is None:
            FedMLSlaveStatusCenter.initialize_status_queue()
        super().__init__(message_queue=self._status_queue, message_center=message_center,
                         is_deployment_status_center=is_deployment_status_center)

    def run_status_dispatcher(self, process_name=None):

        if process_name is not None:
            setproctitle.setproctitle(process_name)

        if platform.system() != "Windows":
            os.setsid()

        # Init status manager instances
        status_manager_instances = dict()
        job_launch_message_map = dict()

        while True:
            message_entity = None

            # Check if we should stop status dispatcher
            try:
                self.check_status_stop_event()
            except StatusCenterStoppedException as e:
                break

            # Dispatch status messages.
            # noinspection PyBroadException
            try:
                # Get the status message from the queue
                try:
                    message_body = self.status_queue.get(block=False, timeout=0.1)
                except queue.Empty as e:  # If queue is empty, then break loop
                    message_body = None
                if message_body is None:
                    time.sleep(0.1)
                    continue

                # Build message and status entity
                message_entity = FedMLMessageEntity(message_body=message_body)
                status_entity = FedMLStatusEntity(status_msg_body=message_body)

                # Generate status manager instance
                run_id_str = str(status_entity.run_id)
                run_id_int = int(status_entity.run_id)
                if status_manager_instances.get(run_id_str) is None:
                    if len(status_manager_instances.keys()) >= FedMLStatusCenter.ALLOWED_MAX_JOB_STATUS_CACHE_NUM:
                        for iter_run_id, iter_status_mgr in status_manager_instances.items():
                            if iter_status_mgr.is_job_completed():
                                status_manager_instances.pop(iter_run_id)
                                break

                    status_manager_instances[run_id_str] = FedMLStatusManager(
                        run_id=run_id_int, edge_id=status_entity.edge_id, status_center=self,
                        message_center=self.message_center)
                else:
                    status_manager_instances[run_id_str].edge_id = status_entity.edge_id

                # Process the slave status
                if message_entity.topic.startswith(FedMLStatusCenter.TOPIC_SLAVE_STATUS_PREFIX):
                    # Report the slave status to master
                    status_manager_instances[run_id_str]. \
                        status_center_process_slave_status_to_master_in_slave_agent(
                        message_entity.topic, message_entity.payload
                    )
                elif message_entity.topic.startswith(FedMLStatusCenter.TOPIC_SLAVE_STATUS_TO_MLOPS_PREFIX):
                    # Report slave status to mlops (Active/IDLE message)
                    status_manager_instances[run_id_str]. \
                        status_center_process_slave_status_to_mlops_in_slave_agent(
                        message_entity.topic, message_entity.payload
                    )
                elif (message_entity.topic.startswith(FedMLStatusCenter.TOPIC_SLAVE_JOB_LAUNCH_PREFIX) and
                      message_entity.topic.endswith(FedMLStatusCenter.TOPIC_SLAVE_JOB_LAUNCH_SUFFIX)):
                    pass
                    # Async request the job status from master when launching the job
                    # job_launch_message_map[run_id_str] = {"topic": message_entity.topic,
                    #                                       "payload": message_entity.payload}
                    # status_manager_instances[run_id_str]. \
                    #     status_center_request_job_status_from_master_in_slave_agent(
                    #     message_entity.topic, message_entity.payload
                    # )
                elif (message_entity.topic.startswith(FedMLStatusCenter.TOPIC_SLAVE_JOB_STOP_PREFIX) and
                      message_entity.topic.endswith(FedMLStatusCenter.TOPIC_SLAVE_JOB_STOP_SUFFIX)):
                    # Cleanup when stopped the job
                    if job_launch_message_map.get(run_id_str, None) is not None:
                        job_launch_message_map.pop(run_id_str)

            except Exception as e:
                if message_entity is not None:
                    logging.info(
                        f"Failed to process the status with topic {message_entity.topic}, "
                        f"payload {message_entity.payload}, {traceback.format_exc()}")
                else:
                    logging.info(f"Failed to process the status: {traceback.format_exc()}")
