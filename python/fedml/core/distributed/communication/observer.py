from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def receive_message(self, msg_type, msg_params) -> None:
        pass
