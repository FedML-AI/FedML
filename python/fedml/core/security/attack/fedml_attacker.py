from .attack_method_sample_a import AttackMethodA
from ..constants import ATTACK_METHOD_A


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
        if hasattr(args, "enable_attack") and args.enable_attack:
            self.is_enabled = True
            self.attack_type = args.attack_type.strip()
            self.attacker = None
            if self.attack_type == ATTACK_METHOD_A:
                self.attacker = AttackMethodA()
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

    def attack(self, local_w, global_w, refs=None):
        return self.attacker.attack(local_w, global_w, refs)
