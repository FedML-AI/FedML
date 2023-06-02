import logging
from abc import ABC, abstractmethod
from ..security.fedml_attacker import FedMLAttacker
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy


class ClientTrainer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class can be used in both server and client side
    3. This class is an operator which does not cache any states inside.
    """

    def __init__(self, model, args):
        self.model = model
        self.id = 0
        self.args = args
        self.local_train_dataset = None
        self.local_test_dataset = None
        self.local_sample_number = 0
        FedMLDifferentialPrivacy.get_instance().init(args)
        FedMLAttacker.get_instance().init(args)

    def set_id(self, trainer_id):
        self.id = trainer_id

    def is_main_process(self):
        return True

    def update_dataset(self, local_train_dataset, local_test_dataset, local_sample_number):
        if FedMLAttacker.get_instance().is_data_poisoning_attack() and FedMLAttacker.get_instance().is_to_poison_data():
            self.local_train_dataset = FedMLAttacker.get_instance().poison_data(local_train_dataset)
            self.local_test_dataset = FedMLAttacker.get_instance().poison_data(local_test_dataset)
        else:
            self.local_train_dataset = local_train_dataset
            self.local_test_dataset = local_test_dataset
        self.local_sample_number = local_sample_number

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    def on_before_local_training(self, train_data, device, args):
        pass

    @abstractmethod
    def train(self, train_data, device, args):
        pass

    def on_after_local_training(self, train_data, device, args):
        if FedMLDifferentialPrivacy.get_instance().is_local_dp_enabled():
            logging.info("-----add local DP noise ----")
            model_params_with_dp_noise = FedMLDifferentialPrivacy.get_instance().add_local_noise(self.get_model_params())
            self.set_model_params(model_params_with_dp_noise)

    def test(self, test_data, device, args):
        pass
