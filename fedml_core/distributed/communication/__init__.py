from fedml_core.distributed.communication.mpi.com_manager import MpiCommunicationManager
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.communication.observer import Observer
from abc import ABC, abstractmethod


class CommunicationManager(ABC):
    @abstractmethod
    def send(self, msg):
        pass

    @abstractmethod
    def add_observer(self, observer: Observer):
        pass

    @abstractmethod
    def remove_observer(self, observer: Observer):
        pass
