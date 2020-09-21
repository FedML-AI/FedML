from abc import abstractmethod

from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.communication.observer import Observer


class BaseCommunicationManager(object):

    @abstractmethod
    def send_message(self, msg: Message):
        pass

    @abstractmethod
    def add_observer(self, observer: Observer):
        pass

    @abstractmethod
    def remove_observer(self, observer: Observer):
        pass

    @abstractmethod
    def handle_receive_message(self):
        pass

    @abstractmethod
    def stop_receive_message(self):
        pass
