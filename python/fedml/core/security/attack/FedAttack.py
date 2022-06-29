from abc import ABC, abstractmethod


class FedAttack(ABC):
    @abstractmethod
    def attack(self, local_w, global_w, refs=None):
        pass
