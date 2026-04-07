import logging

import numpy as np
import torch
from torch import nn

from fedml.core import ClientTrainer


class VeriFLTrainer(ClientTrainer):
    def __init__(self, model, args):
        self.cpu_transfer = bool(getattr(args, "cpu_transfer", True))
        super().__init__(model, args)

        # --- Scaling (model_replacement) backdoor training config ---
        self._scaling_attack_enabled = (
            bool(getattr(args, "enable_attack", False))
            and str(getattr(args, "attack_type", "")).strip() == "model_replacement"
        )
        if self._scaling_attack_enabled:
            self._byzantine_client_num = int(getattr(args, "byzantine_client_num", 0))
            atr = getattr(args, "attack_training_rounds", None)
            self._attack_training_rounds = atr if isinstance(atr, list) else None
            self._backdoor_per_batch = int(getattr(args, "backdoor_per_batch", 20))
            self._target_label = int(getattr(args, "target_label", 0))
            self._trigger_size = int(getattr(args, "trigger_size", 3))
            self._trigger_value = float(getattr(args, "trigger_value", 1.0))
            self._random_seed = int(getattr(args, "random_seed", 0))
        else:
            self._byzantine_client_num = 0
        self._current_round = 0
        self._scaling_logged_init = False
        self._scaling_rng = None

    def get_model_params(self):
        if self.cpu_transfer:
            return self.model.cpu().state_dict()
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def _ensure_scaling_rng(self):
        """Create isolated RNG for this malicious client (called once)."""
        if self._scaling_rng is None:
            rng_seed = self._random_seed + self.id + 2**20
            self._scaling_rng = np.random.default_rng(rng_seed)
            logging.info(
                "Scaling backdoor RNG | client_id=%d | rng_seed=%d",
                self.id, rng_seed,
            )

    def train(self, train_data, device, args):
        # ------------------------------------------------------------------
        # F-6 fix: FedMLTrainer.train() always passes self.train_local (the
        # *original* clean DataLoader).  ClientTrainer.update_dataset() stores
        # the poisoned DataLoader in self.local_train_dataset, but train()
        # never reads it.  Redirect here so data-poisoning attacks (Label
        # Flipping) actually affect training.
        #
        # Safety note (E-10): after update_dataset(), local_train_dataset is
        # non-None for ALL clients (including benign ones).  For benign
        # clients, local_train_dataset == the clean DataLoader passed in,
        # so the redirect is a no-op (same object reference).
        # ------------------------------------------------------------------
        if hasattr(self, 'local_train_dataset') and self.local_train_dataset is not None:
            train_data = self.local_train_dataset

        # --- Scaling attack: determine if this client poisons this round ---
        is_malicious = (
            self._scaling_attack_enabled
            and self.id < self._byzantine_client_num
        )
        poison_this_round = (
            is_malicious
            and self._attack_training_rounds is not None
            and self._current_round in self._attack_training_rounds
        )

        # Lazy init log (printed once per client)
        if self._scaling_attack_enabled and not self._scaling_logged_init:
            logging.info(
                "Scaling backdoor init | client_id=%d | is_malicious=%s "
                "| backdoor_per_batch=%d | target_label=%d "
                "| trigger_size=%d | trigger_value=%s",
                self.id,
                is_malicious,
                self._backdoor_per_batch,
                self._target_label,
                self._trigger_size,
                self._trigger_value,
            )
            self._scaling_logged_init = True

        if poison_this_round:
            self._ensure_scaling_rng()

        # --- Standard training setup ---
        model = self.model
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss().to(device)

        # Optionally use attacker-specific hyperparams
        if is_malicious:
            attacker_lr = getattr(args, "attacker_lr", None)
            attacker_wd = getattr(args, "attacker_weight_decay", None)
            lr = float(attacker_lr if attacker_lr is not None else getattr(args, "learning_rate", 0.01))
            wd = float(attacker_wd if attacker_wd is not None else getattr(args, "weight_decay", 0.0))
        else:
            lr = float(getattr(args, "learning_rate", 0.01))
            wd = float(getattr(args, "weight_decay", 0.0))

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(getattr(args, "momentum", 0.9)),
            weight_decay=wd,
        )
        logging.info(
            "Client %s optimizer | lr=%s momentum=%s weight_decay=%s",
            self.id,
            optimizer.defaults["lr"],
            optimizer.defaults["momentum"],
            optimizer.defaults["weight_decay"],
        )

        # Epoch count
        if is_malicious:
            attacker_epochs = getattr(args, "attacker_epochs", None)
            num_epochs = int(attacker_epochs if attacker_epochs is not None else getattr(args, "epochs", 1))
        else:
            num_epochs = int(getattr(args, "epochs", 1))

        epoch_loss = []
        total_poisoned_samples = 0
        total_poisoned_batches = 0

        for epoch in range(num_epochs):
            batch_losses = []
            for batch_idx, (images, labels) in enumerate(train_data):
                images, labels = images.to(device), labels.to(device)

                # --- Backdoor injection (only for malicious clients in attack rounds) ---
                if poison_this_round:
                    batch_size = images.size(0)
                    inject_count = min(self._backdoor_per_batch, batch_size)
                    indices = self._scaling_rng.choice(
                        batch_size, size=inject_count, replace=False
                    )
                    images[indices, :, -self._trigger_size:, -self._trigger_size:] = self._trigger_value
                    labels[indices] = self._target_label
                    total_poisoned_samples += inject_count
                    total_poisoned_batches += 1

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

                if batch_idx % 20 == 0:
                    logging.info(
                        "Client %s | Epoch %s | Batch %s/%s | Loss %.6f",
                        self.id,
                        epoch + 1,
                        batch_idx + 1,
                        len(train_data),
                        loss.item(),
                    )
            if batch_losses:
                epoch_loss.append(sum(batch_losses) / len(batch_losses))

        if epoch_loss:
            logging.info(
                "Client %s finished local training with mean loss %.6f",
                self.id,
                sum(epoch_loss) / len(epoch_loss),
            )

        # --- Scaling backdoor epoch summary ---
        if poison_this_round:
            logging.info(
                "Scaling backdoor epoch summary | client_id=%d | round=%d "
                "| poisoned_samples=%d | poisoned_batches=%d",
                self.id,
                self._current_round,
                total_poisoned_samples,
                total_poisoned_batches,
            )

        self._current_round += 1

    def test(self, test_data, device, args):
        if test_data is None:
            return None
        model = self.model
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        metrics = {
            "test_correct": 0,
            "test_loss": 0.0,
            "test_total": 0,
        }
        with torch.no_grad():
            for images, labels in test_data:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                _, predicted = torch.max(logits, 1)
                metrics["test_correct"] += predicted.eq(labels).sum().item()
                metrics["test_loss"] += loss.item() * labels.size(0)
                metrics["test_total"] += labels.size(0)
        return metrics
