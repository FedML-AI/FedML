from .attack.byzantine_attack import ByzantineAttack
from .constants import ATTACK_METHOD_BYZANTINE_ATTACK
import logging
from ..common.ml_engine_backend import MLEngineBackend


class FedMLAttacker:
    _attacker_instance = None

    @staticmethod
    def get_instance():
        if FedMLAttacker._attacker_instance is None:
            FedMLAttacker._attacker_instance = FedMLAttacker()

        return FedMLAttacker._attacker_instance

    def __init__(self):
        self.is_enabled = False
        self.attack_type = None
        self.attacker = None

    def init(self, args):
        if hasattr(args, MLEngineBackend.ml_engine_args_flag) and args.ml_engine in [
            MLEngineBackend.ml_engine_backend_tf,
            MLEngineBackend.ml_engine_backend_jax,
            MLEngineBackend.ml_engine_backend_mxnet,
        ]:
            raise ValueError(
                "attacker not supported for this machine learning engine: %s"
                % args.ml_engine
            )

        if hasattr(args, "enable_attack") and args.enable_attack:
            logging.info("------init attack..." + args.attack_type.strip())
            self.is_enabled = True
            self.attack_type = args.attack_type.strip()
            self.attacker = None
            if self.attack_type == ATTACK_METHOD_BYZANTINE_ATTACK:
                self.attacker = ByzantineAttack(args)
            # elif self.attack_type == ATTACK_METHOD_DLG:
            #     self.attacker = DLGAttack(model=args.model, attack_epoch=args.attack_epoch)
        else:
            self.is_enabled = False

    def is_attack_enabled(self):
        return self.is_enabled

    def get_attack_types(self):
        return self.attack_type

    def is_server_attack(self, attack_type):
        pass

    def is_client_attack(self, attack_type):
        pass

    def is_model_attack(self):
        if self.is_attack_enabled() and self.attack_type in [
            ATTACK_METHOD_BYZANTINE_ATTACK
        ]:
            return True
        return False

    def attack_model(self, local_w, global_w, refs=None):
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        return self.attacker.attack_model(local_w, global_w, refs)

    def poison_data(self, dataset):
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        return self.attacker.poison_data(dataset)

    def reconstruct(self, local_w, global_w, refs=None):
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        return self.attacker.reconstruct(local_w, global_w, refs=None)
