import logging
from enum import Enum, unique
import multiprocessing

import fedml
from abc import ABC, abstractmethod
from .message_common import FedMLMessageEntity


@unique
class JobStatus(Enum):
    STATUS_OFFLINE = "OFFLINE"
    STATUS_PROVISIONING = "PROVISIONING"
    STATUS_IDLE = "IDLE"
    UPGRADING = "UPGRADING"
    STARTING = "STARTING"
    STATUS_RUNNING = "RUNNING"
    STATUS_STOPPING = "STOPPING"
    STATUS_KILLED = "KILLED"
    STATUS_FAILED = "FAILED"
    STATUS_FINISHED = "FINISHED"
    STATUS_EXCEPTION = "EXCEPTION"

    def __str__(self):
        return self.value

    @classmethod
    def get_job_enum_from_str(cls, job_status_str: str):
        for job_status in cls:
            if job_status.value == job_status_str:
                return job_status
        return cls.STATUS_OFFLINE

    @staticmethod
    def is_job_completed(job_status_str: str):
        if job_status_str == JobStatus.STATUS_FINISHED.value or \
                job_status_str == JobStatus.STATUS_FAILED.value or \
                job_status_str == JobStatus.STATUS_KILLED.value or \
                job_status_str == JobStatus.STATUS_EXCEPTION.value:
            return True

        return False


@unique
class DeviceStatus(Enum):
    STATUS_OFFLINE = "OFFLINE"
    STATUS_PROVISIONING = "PROVISIONING"
    STATUS_IDLE = "IDLE"
    STATUS_UPGRADING = "UPGRADING"
    STATUS_QUEUED = "QUEUED"
    STATUS_INITIALIZING = "INITIALIZING"
    STATUS_TRAINING = "TRAINING"
    STATUS_RUNNING = "RUNNING"
    STATUS_STOPPING = "STOPPING"
    STATUS_KILLED = "KILLED"
    STATUS_FAILED = "FAILED"
    STATUS_EXCEPTION = "EXCEPTION"
    STATUS_FINISHED = "FINISHED"

    def __str__(self):
        return self.value

    @classmethod
    def get_device_enum_from_str(cls, device_status_str: str):
        for device_status in cls:
            if device_status.value == device_status_str:
                return device_status
        return cls.STATUS_OFFLINE


class FedMLStatusCenter(ABC):
    TOPIC_MASTER_STATUS_PREFIX = "fl_server/flserver_agent_"
    TOPIC_SLAVE_STATUS_PREFIX = "fl_client/flclient_agent_"
    TOPIC_SLAVE_STATUS_TO_MLOPS_PREFIX = "fl_run/fl_client/mlops/status"
    TOPIC_SLAVE_JOB_LAUNCH_PREFIX = "flserver_agent/"
    TOPIC_SLAVE_JOB_LAUNCH_SUFFIX = "/start_train"
    TOPIC_SLAVE_JOB_STOP_PREFIX = "flserver_agent/"
    TOPIC_SLAVE_JOB_STOP_SUFFIX = "/stop_train"
    ALLOWED_MAX_JOB_STATUS_CACHE_NUM = 1000

    def __init__(self, message_queue, message_center, is_deployment_status_center):
        self.status_queue = message_queue
        self.message_center = message_center
        self.is_deployment_status_center = is_deployment_status_center
        self.status_event = multiprocessing.Event()
        self.status_center_process = None
        self.status_manager_instance = None
        self.status_sender_message_center_queue = None
        self.status_listener_message_center_queue = None
        self.status_runner = None
        self.start_status_center()

    @abstractmethod
    def run_status_dispatcher(self):
        pass

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def start_status_center(self):
        if self.status_center_process is None:
            self.status_event.clear()
            self.status_center_process = fedml.get_process(
                target=self.run_status_dispatcher)

            self.status_center_process.start()

    def stop_status_center(self):
        if self.status_event is not None:
            self.status_event.set()

    def check_status_stop_event(self):
        if self.status_event is not None and self.status_event.is_set():
            logging.info("Received status center stopping event.")
            raise StatusCenterStoppedException("Status center stopped (for sender)")

    def send_message(self, topic, payload, run_id=None):
        message_entity = FedMLMessageEntity(topic=topic, payload=payload, run_id=run_id)
        self.status_queue.put(message_entity.get_message_body())

    def send_message_json(self, topic, payload):
        self.send_message(topic, payload)

    def send_status_message(self, topic, payload):
        message_entity = FedMLMessageEntity(topic=topic, payload=payload)
        self.status_queue.put(message_entity.get_message_body())

    def get_status_queue(self):
        return self.status_queue

    def set_status_queue(self, status_queue):
        self.status_queue = status_queue

    # TODO (alaydshah): Remove this method if not needed
    def status_center_process_master_status(self, topic, payload):
        pass

    # TODO (alaydshah): Remove this method if not needed
    def status_center_process_slave_status(self, topic, payload):
        pass

    def rebuild_message_center(self, message_center_queue):
        pass

    def rebuild_status_center(self, status_queue):
        pass

    def register_job_launch_message(self, topic, payload):
        message_entity = FedMLMessageEntity(topic=topic, payload=payload)
        self.status_queue.put(message_entity.get_message_body())

    def register_job_stop_message(self, topic, payload):
        message_entity = FedMLMessageEntity(topic=topic, payload=payload)
        self.status_queue.put(message_entity.get_message_body())

    @staticmethod
    def rebuild_status_center_from_queue(status_queue):
        status_center = FedMLStatusCenter(message_queue=status_queue)
        return status_center


class StatusCenterStoppedException(Exception):
    """ Status center stopped. """
    pass
