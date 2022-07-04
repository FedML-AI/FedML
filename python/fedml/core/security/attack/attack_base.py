from abc import ABC, abstractmethod


class BaseAttackMethod(ABC):
    @abstractmethod
    def attack(self, local_w, global_w, refs=None):
        pass
