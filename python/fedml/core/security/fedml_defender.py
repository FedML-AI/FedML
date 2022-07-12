import logging
from ...core.security.defense.norm_diff_clipping_defense import NormDiffClippingDefense
from ...core.security.constants import DEFENSE_DIFF_CLIPPING


class FedMLDefender:

    _defender_instance = None

    @staticmethod
    def get_instance():
        if FedMLDefender._defender_instance is None:
            FedMLDefender._defender_instance = FedMLDefender()

        return FedMLDefender._defender_instance

    def __init__(self):
        self.is_enabled = False
        self.defense_type = None
        self.defender = None

    def init(self, args):
        if hasattr(args, "enable_defense") and args.enable_defense:
            self.is_enabled = True
            self.defense_type = args.defense_type.strip()
            logging.info("self.defense_type = {}".format(self.defense_type))
            self.defender = None
            if self.defense_type == DEFENSE_DIFF_CLIPPING:
                self.defender = NormDiffClippingDefense(args.norm_bound)
            else:
                raise Exception("args.attack_type is not defined!")
        else:
            self.is_enabled = False

    def is_defense_enabled(self):
        return self.is_enabled

    def get_defense_types(self):
        return self.defense_type

    def is_server_defense(self, defense_type):
        return self.is_enabled and defense_type in [""]

    def is_client_defense(self, defense_type):
        return self.is_enabled and defense_type in [DEFENSE_DIFF_CLIPPING]

    def defend(self, local_w, global_w, refs=None):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        return self.defender.defend(local_w, global_w, refs)
