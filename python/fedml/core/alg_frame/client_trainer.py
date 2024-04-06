import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..security.fedml_attacker import FedMLAttacker
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ..fhe.fhe_agg import FedMLFHE


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
        self.rid = 0
        self.template_model_params: Optional[Any] = None
        self.enc_model_params = None
        FedMLDifferentialPrivacy.get_instance().init(args)
        FedMLAttacker.get_instance().init(args)
        FedMLFHE.get_instance().init(args)

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

    def get_enc_model_params(self):
        return self.enc_model_params

    def set_enc_model_params(self, enc_model_parameters):
        self.enc_model_params = enc_model_parameters

    def on_before_local_training(self, train_data, device, args):
        if FedMLFHE.get_instance().is_fhe_enabled():
            if self.rid != 0:
                if self.template_model_params is None:
                    self.template_model_params = self.get_model_params()

                # get encrypted global params, and decrypt then set params
                logging.info(" ---- decrypting aggregated model ----")
                dec_aggregated_model = FedMLFHE.get_instance().fhe_dec(
                    self.template_model_params,
                    self.get_enc_model_params()
                )
                self.set_model_params(dec_aggregated_model)
            self.rid += 1

    @abstractmethod
    def train(self, train_data, device, args):
        pass

    def on_after_local_training(self, train_data, device, args):
        if FedMLFHE.get_instance().is_fhe_enabled():
            # encrypt before sending to server
            logging.info(" ---- encrypting client model ----")
            encrypted_model_params = FedMLFHE.get_instance().fhe_enc('local', self.get_model_params())
            self.set_enc_model_params(encrypted_model_params)
        elif FedMLDifferentialPrivacy.get_instance().is_local_dp_enabled():
            logging.info("-----add local DP noise ----")
            model_params_with_dp_noise = FedMLDifferentialPrivacy.get_instance().add_local_noise(
                self.get_model_params()
            )
            self.set_model_params(model_params_with_dp_noise)

    def test(self, test_data, device, args):
        pass
