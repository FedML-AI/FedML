from .attack.byzantine_attack import ByzantineAttack
from .attack.dlg_attack import DLGAttack
from .attack.label_flipping_attack import LabelFlippingAttack
from .attack.model_replacement_backdoor_attack import ModelReplacementBackdoorAttack
from .constants import ATTACK_METHOD_BYZANTINE_ATTACK, ATTACK_LABEL_FLIPPING, BACKDOOR_ATTACK_MODEL_REPLACEMENT, \
    ATTACK_METHOD_DLG
import logging
from ..common.ml_engine_backend import MLEngineBackend
from typing import List, Tuple, Any
from collections import OrderedDict


class FedMLAttacker:
    """
    Represents an attacker in a federated learning system.

    The `FedMLAttacker` class is responsible for managing different types of attacks, including model poisoning, data poisoning,
    and data reconstruction attacks, within a federated learning setting.

    Attributes:
        _attacker_instance (FedMLAttacker): A singleton instance of the `FedMLAttacker` class.
        is_enabled (bool): Whether the attacker is enabled.
        attack_type (str): The type of attack being used.
        attacker (Any): The specific attacker object.
    """
    _attacker_instance = None

    @staticmethod
    def get_instance():
        if FedMLAttacker._attacker_instance is None:
            FedMLAttacker._attacker_instance = FedMLAttacker()

        return FedMLAttacker._attacker_instance

    def __init__(self):
        """
        Initialize a FedMLAttacker instance.

        This constructor sets up the attacker instance and initializes its properties.

        Attributes:
            is_enabled (bool): Whether the attacker is enabled.
            attack_type (str): The type of attack being used.
            attacker (Any): The specific attacker object.

        """
        self.is_enabled = False
        self.attack_type = None
        self.attacker = None

    def init(self, args):
        """
        Initialize the attacker with provided arguments.

        This method initializes the attacker based on the provided arguments.

        Args:
            args: The arguments used to configure the attacker.

        """
        if hasattr(args, "enable_attack") and args.enable_attack:
            logging.info("------init attack..." + args.attack_type.strip())
            self.is_enabled = True
            self.attack_type = args.attack_type.strip()
            self.attacker = None
            if self.attack_type == ATTACK_METHOD_BYZANTINE_ATTACK:
                self.attacker = ByzantineAttack(args)
            elif self.attack_type == ATTACK_LABEL_FLIPPING:
                self.attacker = LabelFlippingAttack(args)
            elif self.attack_type == BACKDOOR_ATTACK_MODEL_REPLACEMENT:
                self.attacker = ModelReplacementBackdoorAttack(args)
            elif self.attack_type == ATTACK_METHOD_DLG:
                self.attacker = DLGAttack(args=args)
        else:
            self.is_enabled = False

        if self.is_enabled:
            if hasattr(args, MLEngineBackend.ml_engine_args_flag) and args.ml_engine in [
                MLEngineBackend.ml_engine_backend_tf,
                MLEngineBackend.ml_engine_backend_jax,
                MLEngineBackend.ml_engine_backend_mxnet,
            ]:
                logging.info(
                    "FedMLAttacker is not supported for the machine learning engine: %s. "
                    "We will support more engines in the future iteration."
                    % args.ml_engine
                )
                self.is_enabled = False

    def is_attack_enabled(self):
        """
        Check if the attacker is enabled.

        Returns:
            bool: True if the attacker is enabled, False otherwise.

        """
        return self.is_enabled

    def get_attack_types(self):
        """
        Get the type of attack.

        Returns:
            str: The type of attack being used.

        """
        return self.attack_type

    # --------------- for model poisoning attacks --------------- #
    def is_model_attack(self):
        """
        Check if the attack is a model poisoning attack.

        Returns:
            bool: True if it's a model poisoning attack, False otherwise.

        """

        if self.is_attack_enabled() and self.attack_type in [
            ATTACK_METHOD_BYZANTINE_ATTACK, BACKDOOR_ATTACK_MODEL_REPLACEMENT
        ]:
            return True
        return False

    def attack_model(self, raw_client_grad_list: List[Tuple[float, OrderedDict]], extra_auxiliary_info: Any = None):
        """
        Attack the model with poisoned gradients.

        This method is used for model poisoning attacks.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]): A list of tuples, each containing a weight and a
                dictionary of client gradients.
            extra_auxiliary_info (Any, optional): Additional auxiliary information for the attack.

        Returns:
            Any: The poisoned client gradients.

        Raises:
            Exception: If the attacker is not initialized.

        """
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        return self.attacker.attack_model(raw_client_grad_list, extra_auxiliary_info)
    # --------------- for model poisoning attacks --------------- #

    # --------------- for data poisoning attacks --------------- #
    def is_data_poisoning_attack(self):
        """
        Check if the attack is a data poisoning attack.

        Returns:
            bool: True if it's a data poisoning attack, False otherwise.

        """
        if self.is_attack_enabled() and self.attack_type in [ATTACK_LABEL_FLIPPING]:
            return True
        return False

    def is_to_poison_data(self):
        """
        Check if data should be poisoned.

        Returns:
            bool: True if data should be poisoned, False otherwise.

        Raises:
            Exception: If the attacker is not initialized.

        """
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        return self.attacker.is_to_poison_data()

    def poison_data(self, dataset):
        """
        Poison the dataset.

        This method is used for data poisoning attacks.

        Args:
            dataset: The dataset to be poisoned.

        Returns:
            Any: The poisoned dataset.

        Raises:
            Exception: If the attacker is not initialized.

        """
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        return self.attacker.poison_data(dataset)
    # --------------- for data poisoning attacks --------------- #

    # --------------- for data reconstructing attacks --------------- #
    def is_data_reconstruction_attack(self):
        """
        Check if the attack is a data reconstruction attack.

        Returns:
            bool: True if it's a data reconstruction attack, False otherwise.

        """
        if self.is_attack_enabled() and self.attack_type in [ATTACK_METHOD_DLG]:
            return True
        return False

    def reconstruct_data(self, raw_client_grad_list: List[Tuple[float, OrderedDict]], extra_auxiliary_info: Any = None):
        """
        Reconstruct the data from gradients.

        This method is used for data reconstruction attacks.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]): A list of tuples, each containing a weight and a
                dictionary of client gradients.
            extra_auxiliary_info (Any, optional): Additional auxiliary information for the attack.

        Raises:
            Exception: If the attacker is not initialized.

        """
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        self.attacker.reconstruct_data(
            raw_client_grad_list, extra_auxiliary_info=extra_auxiliary_info)
    # --------------- for data reconstructing attacks --------------- #
