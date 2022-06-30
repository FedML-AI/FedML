class FedMLAttacker:
    def __init__(self, args):
        if hasattr(args, "enable_attack") and args.enable_attack == "Y":
            self.is_enabled = True
            self.attacks = {}
            for attack in args.attack_type.split(","):
                pass
                # if attack.trim == "x":
                #     self.attacks["x"] = X(args.norm_bound)
        else:
            self.is_enabled = False

    def is_attack_enabled(self):
        return self.is_enabled

    def is_server_attack(self, attack_type):
        return self.is_enabled and attack_type in [""]

    def is_client_attack(self, attack_type):
        return self.is_enabled and attack_type in [""]

    def attack(self, attack_type, local_w, global_w, refs=None):
        return self.attacks[attack_type].attack(local_w, global_w, refs)
