import logging
import multiprocessing
import os
import platform
import queue
import time

from fedml.computing.scheduler.scheduler_core.message_common import FedMLMessageEntity, FedMLStatusEntity
from fedml.computing.scheduler.scheduler_core.status_center import FedMLStatusCenter, StatusCenterStoppedException


class FedMLMasterStatusCenter(FedMLStatusCenter):
    _status_queue = None

    @classmethod
    def initialize_status_queue(cls):
        if cls._status_queue is None:
            cls._status_queue = multiprocessing.Manager().Queue(-1)  # or any other appropriate initialization

    def __init__(self, message_center, is_deployment_status_center):
        if FedMLMasterStatusCenter._status_queue is None:
            FedMLMasterStatusCenter.initialize_status_queue()
        super().__init__(message_queue=self._status_queue, message_center=message_center,
                         is_deployment_status_center=is_deployment_status_center)

    def run_status_dispatcher(self):
        if platform.system() != "Windows":
            os.setsid()

        # Init status manager instances
        status_manager_instances = dict()

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
                        run_id=run_id_int, edge_id=status_entity.edge_id,
                        server_id=status_entity.server_id, status_center=self,
                        message_center=message_center)
                else:
                    status_manager_instances[run_id_str].edge_id = status_entity.edge_id
                    if status_entity.server_id is not None and str(status_entity.server_id) != "0":
                        status_manager_instances[run_id_str].server_id = status_entity.server_id

                # if the job status is completed then continue
                if status_manager_instances[run_id_str].is_job_completed():
                    continue

                # Process the master and slave status.
                if message_entity.topic.startswith(FedMLStatusCenter.TOPIC_MASTER_STATUS_PREFIX):
                    # Process the job status
                    status_manager_instances[run_id_str].status_center_process_master_status(
                        message_entity.topic, message_entity.payload)

                    # Save the job status
                    status_manager_instances[run_id_str].save_job_status()

                elif message_entity.topic.startswith(FedMLStatusCenter.TOPIC_SLAVE_STATUS_PREFIX):
                    # Process the slave device status
                    status_manager_instances[run_id_str].status_center_process_slave_status(
                        message_entity.topic, message_entity.payload)

                    # Save the device status in job
                    status_manager_instances[run_id_str].save_device_status_in_job(status_entity.edge_id)

            except Exception as e:
                if message_entity is not None:
                    logging.info(
                        f"Failed to process the status with topic {message_entity.topic}, "
                        f"payload {message_entity.payload}, {traceback.format_exc()}")
                else:
                    logging.info(f"Failed to process the status: {traceback.format_exc()}")
