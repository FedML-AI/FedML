from enum import Enum
from abc import ABC, abstractmethod


# Define an enum for job status
class JobStatus(Enum):
    """
    Enum for job status
    """
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class Job(ABC):
    def __init__(self, name):
        """
        Initialize the Job instance.

        Parameters:
        - name (str): Name for the job. This is used to identify the job in the workflow so it should be unique.
        """
        self.name = name

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
    def status(self):
        """
        Abstract method to get the status of the job.
        Represents the status of the job, which should be of type JobStatus: running, completed, or error.
        """

    @abstractmethod
    def kill(self):
        """
        Method to kill the job if running on remote server.
        """
