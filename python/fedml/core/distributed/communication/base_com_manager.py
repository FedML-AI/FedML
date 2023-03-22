from abc import abstractmethod

from .message import Message
from .observer import Observer


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
