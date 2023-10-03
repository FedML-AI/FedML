from typing import Any, Dict, List, Optional, Tuple, Union

from torch.optim import Optimizer
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .hf_trainer import HFTrainer


class HFResumeTrainerCallback(TrainerCallback):
    def __init__(self, reset_list: Optional[List[Tuple[Optional[Any], str, Any]]] = None):
        """

        Args:
            reset_list: a list of (object, attribute_name, value_to_recover) for recovery on_train_begin
        """
        if reset_list is None:
            reset_list = []
        self.reset_list = reset_list

    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        for (obj, attr_name, value) in self.reset_list:
            if obj is not None and hasattr(obj, attr_name):
                setattr(obj, attr_name, value)
        self.reset_list.clear()

        return control


class HFResumeTrainer(HFTrainer):
    def __init__(
            self,
            *pos_args,
            is_resume_train: bool = False,
            resume_train_callback: Optional[HFResumeTrainerCallback] = None,
            **kwargs
    ):
        super().__init__(*pos_args, **kwargs)

        # set to `True` if continuing from previous early-stopped train
        self.is_resume_train = is_resume_train

        if resume_train_callback is None:
            resume_train_callback = HFResumeTrainerCallback()
        self.resume_train_callback = resume_train_callback
        self.add_callback(self.resume_train_callback)

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        if not self.is_resume_train:
            return super().create_optimizer_and_scheduler(num_training_steps)

    def create_optimizer(self) -> Optimizer:
        if not self.is_resume_train:
            return super().create_optimizer()
        else:
            return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Optimizer = None):
        if not self.is_resume_train:
            return super().create_scheduler(num_training_steps, optimizer)
        else:
            return self.lr_scheduler

    def dummy_function(self, *args, **kwargs) -> None:
        return None

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs
    ):
        if self.is_resume_train:
            reset_list = self.resume_train_callback.reset_list

            # turn off TrainingArguments.deepspeed to avoid duplicated initializations
            # TODO: verify model, model_wrapped, deepspeed, optimizer, lr_scheduler after reset
            reset_list.append((self.args, "deepspeed", self.args.deepspeed))
            self.args.deepspeed = None

            if hasattr(self, "accelerator"):
                # when resuming, should disable the free_memory function call at the beginning
                # of Trainer._inner_training_loop
                reset_list.append((self.accelerator, "free_memory", self.accelerator.free_memory))
                self.accelerator.free_memory = self.dummy_function

            if hasattr(self, "is_deepspeed_enabled"):
                reset_list.append((self, "is_deepspeed_enabled", self.is_deepspeed_enabled))
                self.is_deepspeed_enabled = False

        super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )

        self.is_resume_train = True
