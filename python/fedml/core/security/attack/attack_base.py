from abc import ABC, abstractmethod


class BaseAttackMethod(ABC):
    @abstractmethod
    def attack_model(self, local_w, global_w, refs=None):
        pass

    @abstractmethod
    def attack_data(self, train_data_loader, test_data_loader):
        pass
