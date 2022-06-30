from fedml.core.security.defense.norm_diff_clipping import NormDiffClipping

DIFF_CLIPPING = "norm_diff_clipping"


class FedMLDefenser:
    def __init__(self, args):
        if hasattr(args, "enable_defense") and args.enable_defense == "Y":
            self.is_enabled = True
            self.defenses = {}
            for defense in args.defense_type.split(","):
                if defense.strip() == DIFF_CLIPPING:
                    self.defenses[DIFF_CLIPPING] = NormDiffClipping(
                        args.norm_bound
                    )
        else:
            self.is_enabled = False

    def is_defense_enabled(self):
        return self.is_enabled

    def get_defense_types(self):
        return self.defenses.keys()

    def is_server_defense(self, defense_type):
        return self.is_enabled and defense_type in [""]

    def is_client_defense(self, defense_type):
        return self.is_enabled and defense_type in [DIFF_CLIPPING]

    def defense(self, defense_type, local_w, global_w, refs=None):
        return self.defenses[defense_type].defense(local_w, global_w, refs)
