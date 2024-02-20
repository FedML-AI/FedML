from enum import Enum
from abc import ABC, abstractmethod
from fedml.api.constants import RunStatus
from typing import List, Dict, Any
import argparse


# Define an enum for job status
class JobStatus(Enum):

    """
    Enum for job status
    """
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    UNDETERMINED = "UNDETERMINED"

    @classmethod
    def _create_run_status_to_job_status_mapping(cls):
        cls._run_status_to_job_status_mapping = {
            JobStatus.PROVISIONING: {RunStatus.NOT_STARTED, RunStatus.QUEUED, RunStatus.CLUSTER_QUEUE,
                                     RunStatus.PRE_QUEUE, RunStatus.PROVISIONING},
            JobStatus.RUNNING: {RunStatus.STARTING, RunStatus.RUNNING, RunStatus.LAUNCHED},
            JobStatus.FINISHED: {RunStatus.FINISHED},
            JobStatus.FAILED: {RunStatus.STOPPING, RunStatus.KILLED, RunStatus.FAILED, RunStatus.ABANDONED,
                               RunStatus.ERROR, RunStatus.BLOCKED, RunStatus.INVALID},
            JobStatus.UNDETERMINED: {RunStatus.UNDETERMINED}
        }

    @classmethod
    def get_job_status_from_run_status(cls, run_status: RunStatus):
        if not hasattr(cls, "_run_status_to_job_status_mapping"):
            cls._create_run_status_to_job_status_mapping()
        for job_status, run_status_set in cls._run_status_to_job_status_mapping.items():
            if run_status in run_status_set:
                return job_status
        return JobStatus.UNDETERMINED


class Job(ABC):

    def __init__(self, name: str):
        """
        Initialize the Job instance.

        Parameters:
        - name (str): Name for the job. This is used to identify the job in the workflow so it should be unique.
        """
        self.name = name
        self.input_data_dict: Dict[Any, Any] = dict()
        self.output_data_dict: Dict[Any, Any] = dict()

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @abstractmethod
    def run(self):
        """
        Abstract method to run the job. This method should contain the execution logic of the job.
        """

    @abstractmethod
    def status(self) -> JobStatus:
        """
        Abstract method to get the status of the job.
        Represents the status of the job, which should be of type JobStatus: Running, Success, or Failed.
        """

    @abstractmethod
    def kill(self):
        """
        Method to kill the job if running on remote server.
        """

    def append_input(self, input_job_name: str,  input: Dict):
        """
        Method to append fixed input to the job.
        """
        self.input_data_dict[input_job_name] = input

    def set_inputs(self, inputs: Dict[Any, Any]):
        """
        Method to append fixed input to the job.
        """
        self.input_data_dict = inputs

    def get_outputs(self) -> Dict[Any, Any]:
        """
        Method to get output of the job.
        """
        return self.output_data_dict


class NullJob(Job):
    def __init__(self, name="NullJob"):
        super().__init__(name)

    def run(self):
        self.output_data_dict = self.input_data_dict.copy()

    def status(self):
        return JobStatus.FINISHED

    def kill(self):
        pass