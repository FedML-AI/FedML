from abc import ABC, abstractmethod


class FedDefense(ABC):

    @abstractmethod
    def defense(self, local_w, global_w, refs=None):
        pass
