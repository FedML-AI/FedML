from fedml.core.security.defense.norm_diff_clipping import NormDiffClipping


class FedMLDefenser:
    def __init__(self, args):
        if args.enable_defense == "Y":
            self.is_enabled = True
            self.defenses = {}
            for defense in args.defense_type.split(","):
                if defense.strip() == "norm_diff_clipping":
                    self.defenses["norm_diff_clipping"] = NormDiffClipping(
                        args.norm_bound
                    )
        else:
            self.is_enabled = False

    def is_defense_enabled(self):
        return self.is_enabled

    def is_server_defense(self, defense_type):
        return self.is_enabled and defense_type in ["norm_diff_clipping"]

    def is_client_defense(self, defense_type):
        return self.is_enabled and defense_type in [""]

    def defense(self, defense_type, local_w, global_w, refs=None):
        return self.defenses[defense_type].defense(local_w, global_w, refs)

    def get_defenses_list(self):
        return self.defenses
