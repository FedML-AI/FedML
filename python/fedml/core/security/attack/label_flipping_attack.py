import logging
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from .attack_base import BaseAttackMethod
import numpy as np
from ..common.utils import (
    replace_original_class_with_target_class,
)

"""
ref: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning",
     USENIX Security 2020.
Label Flipping: untargeted data poisoning — flip all labels via a fixed mapping.
attack @client — rewritten to fix D1~D7 per LF_实施规格.md
"""


class LabelFlippingAttack(BaseAttackMethod):
    def __init__(self, args):
        self.original_class_list = args.original_class_list
        self.target_class_list = args.target_class_list
        self.batch_size = args.batch_size

        # --- poison round window ---
        if hasattr(args, "poison_start_round_id") and isinstance(args.poison_start_round_id, int):
            self.poison_start_round_id = args.poison_start_round_id
        else:
            self.poison_start_round_id = 0
        if hasattr(args, "poison_end_round_id") and isinstance(args.poison_end_round_id, int):
            self.poison_end_round_id = args.poison_end_round_id
        else:
            self.poison_end_round_id = args.comm_round - 1

        # --- ratio validation ---
        if hasattr(args, "ratio_of_poisoned_client") and isinstance(args.ratio_of_poisoned_client, float):
            if args.ratio_of_poisoned_client < 0 or args.ratio_of_poisoned_client > 1:
                raise Exception("ratio_of_poisoned_client must be in [0, 1]")
            self.ratio_of_poisoned_client = args.ratio_of_poisoned_client
        else:
            raise Exception("ratio_of_poisoned_client is required")

        # --- W2 fix (D2 + D6): fixed malicious client set using isolated RNG ---
        client_num = args.client_num_in_total
        malicious_num = max(1, math.ceil(client_num * self.ratio_of_poisoned_client))
        random_seed = args.random_seed if hasattr(args, "random_seed") else 0
        # Use isolated RNG so we don't pollute global numpy state (D6 fix)
        rng = np.random.default_rng(seed=random_seed + 2 ** 20)  # offset to decouple from data partition RNG
        if malicious_num >= client_num:
            self.malicious_client_ids = set(range(client_num))
        else:
            chosen = rng.choice(client_num, size=malicious_num, replace=False)
            self.malicious_client_ids = set(int(c) for c in chosen)

        # W8: audit log — print malicious set once at init
        logging.info(
            "[LabelFlippingAttack] Initialized: malicious_client_ids=%s "
            "(count=%d/%d, pmr=%.2f, seed=%d)",
            sorted(self.malicious_client_ids),
            len(self.malicious_client_ids),
            client_num,
            self.ratio_of_poisoned_client,
            random_seed,
        )

        # W7 fix (D7): simple round counter — each MPI process increments once per round
        self.current_round = -1

    # --- W3: accept client_id to decide poisoning ---
    def is_to_poison_data(self, client_id=None, round_idx=None):
        """Return True if this client should poison its data this round."""
        if client_id is None:
            logging.warning("[LabelFlippingAttack] is_to_poison_data called without client_id, defaulting to False")
            return False

        # W7: update round tracking from external source if provided
        if round_idx is not None:
            self.current_round = round_idx
        else:
            self.current_round += 1

        # Check round window
        if self.current_round < self.poison_start_round_id or self.current_round > self.poison_end_round_id:
            return False

        is_malicious = client_id in self.malicious_client_ids
        # W8: per-round audit log
        logging.info(
            "[LabelFlippingAttack] round=%d client_id=%d is_malicious=%s",
            self.current_round, client_id, is_malicious,
        )
        return is_malicious

    def poison_data(self, local_dataset):
        # W4 fix (D3): use long tensor accumulator to preserve label dtype
        tmp_local_dataset_x = torch.Tensor([])
        tmp_local_dataset_y = torch.LongTensor([])
        for batch_idx, (data, targets) in enumerate(local_dataset):
            tmp_local_dataset_x = torch.cat((tmp_local_dataset_x, data))
            tmp_local_dataset_y = torch.cat((tmp_local_dataset_y, targets.long()))

        # Apply label mapping (W1 fix applied in utils.py)
        tmp_y = replace_original_class_with_target_class(
            data_labels=tmp_local_dataset_y,
            original_class_list=self.original_class_list,
            target_class_list=self.target_class_list,
        )
        dataset = TensorDataset(tmp_local_dataset_x, tmp_y)
        # W5 fix (D4): preserve shuffle=True to match original DataLoader
        poisoned_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return poisoned_data