from abc import ABC, abstractmethod


class BaseDefenseMethod(ABC):
    @abstractmethod
    def defend(self, local_w, global_w, refs=None):
        pass
