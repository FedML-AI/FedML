import math

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class PauseResumeCallback(TrainerCallback):
    def __init__(
            self,
            start_global_step: int = -1,
            start_epoch: float = -1.,
            step_threshold: float = math.inf,
            epoch_threshold: float = math.inf
    ):
        if (start_epoch < 0) != (start_global_step < 0):
            raise ValueError(
                f"start_epoch and start_global_step must both be negative or both be non-negative,"
                f" but received start_epoch = {start_epoch}, start_global_step = {start_global_step}."
            )

        self.start_global_step = start_global_step
        self.start_epoch = start_epoch
        self.step_threshold = step_threshold
        self.epoch_threshold = epoch_threshold

    @property
    def use_step_threshold(self) -> bool:
        return not math.isinf(self.step_threshold)

    @property
    def use_epoch_threshold(self) -> bool:
        return not self.use_step_threshold and not math.isinf(self.epoch_threshold)

    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if self.start_global_step < 0 and self.start_epoch < 0:
            # if both values are unset
            self.start_epoch = state.epoch
            self.start_global_step = state.global_step
        else:
            # recover from previous run
            state.epoch = self.start_epoch
            state.global_step = self.start_global_step

        return control

    def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if self.use_step_threshold and state.global_step - self.start_global_step >= self.step_threshold:
            control.should_training_stop = True

        elif self.use_epoch_threshold and state.epoch - self.start_epoch >= self.epoch_threshold:
            # epoch is a float; partial epoch is allowed which means it needs to be checked every step
            control.should_training_stop = True

        return control

    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if state.epoch - self.start_epoch >= self.epoch_threshold:
            control.should_training_stop = True

        return control

    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if args.max_steps > 0:
            # positive `max_steps` overrides `num_train_epochs`
            control.should_training_stop = not bool(state.global_step < args.max_steps)

        elif args.num_train_epochs > 0:
            control.should_training_stop = not bool(state.epoch < args.num_train_epochs)

        # save training progress for resuming
        self.start_global_step = state.global_step
        self.start_epoch = state.epoch

        return control
