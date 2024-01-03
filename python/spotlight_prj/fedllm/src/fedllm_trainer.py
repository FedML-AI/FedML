from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

from fedml.train.llm.configurations import ExperimentArguments
from fedml.train.llm.hf_trainer import HFTrainer
from fedml.train.llm.typing import (
    DataCollatorType,
    DatasetType,
    LrSchedulerType,
    ModelType,
    OptimizerType,
    TokenizerType,
)
from torch import Tensor
from torch.nn import Module
from transformers import (
    EvalPrediction,
    is_optuna_available,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import TrainOutput

if TYPE_CHECKING and is_optuna_available():
    import optuna

from .utils import dummy_func


class FedLLMTrainerCallback(TrainerCallback):
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


class FedLLMTrainer(HFTrainer):
    def __init__(
            self,
            model: Union[ModelType, Module] = None,
            args: ExperimentArguments = None,
            data_collator: Optional[DataCollatorType] = None,
            train_dataset: Optional[DatasetType] = None,
            eval_dataset: Optional[Union[DatasetType, Dict[str, DatasetType]]] = None,
            tokenizer: Optional[TokenizerType] = None,
            model_init: Optional[Callable[[], Union[ModelType, Module]]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[OptimizerType, LrSchedulerType] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
            is_resume_from_interrupt: bool = False,
            resume_train_callback: Optional[FedLLMTrainerCallback] = None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # set to `True` if continuing from previous early-stopped train
        self.is_resume_from_interrupt = is_resume_from_interrupt

        if resume_train_callback is None:
            resume_train_callback = FedLLMTrainerCallback()
        self.resume_train_callback = resume_train_callback
        self.add_callback(self.resume_train_callback)

    def add_callback(self, callback: Union[Type[TrainerCallback], TrainerCallback]) -> None:
        if isinstance(callback, type):
            cb_class = callback
            cb = cb_class()
        else:
            cb_class = type(callback)
            cb = callback

        if issubclass(cb_class, FedLLMTrainerCallback):
            # at most one `FedLLMTrainerCallback` is allowed
            if not self.has_callback(cb) and self.has_callback(FedLLMTrainerCallback):
                self.pop_callback(FedLLMTrainerCallback)
            self.resume_train_callback = cb

        super().add_callback(cb)

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        if not self.is_resume_from_interrupt:
            return super().create_optimizer_and_scheduler(num_training_steps)

    def create_optimizer(self) -> OptimizerType:
        if not self.is_resume_from_interrupt:
            return super().create_optimizer()
        else:
            return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[OptimizerType] = None) -> LrSchedulerType:
        if not self.is_resume_from_interrupt:
            return super().create_scheduler(num_training_steps, optimizer)
        else:
            return self.lr_scheduler

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs
    ) -> TrainOutput:
        if self.is_resume_from_interrupt:
            reset_list = self.resume_train_callback.reset_list

            # turn off TrainingArguments.deepspeed to avoid duplicated initializations
            # TODO: verify model, model_wrapped, deepspeed, optimizer, lr_scheduler after reset
            reset_list.append((self.args, "deepspeed", self.args.deepspeed))
            self.args.deepspeed = None

            reset_list.append((self, "_created_lr_scheduler", self._created_lr_scheduler))
            self._created_lr_scheduler = False

            # when resuming, should disable the free_memory function call at the beginning
            # of Trainer._inner_training_loop
            reset_list.append((self.accelerator, "free_memory", self.accelerator.free_memory))
            self.accelerator.free_memory = dummy_func

            reset_list.append((self, "is_deepspeed_enabled", self.is_deepspeed_enabled))
            self.is_deepspeed_enabled = False

        train_output = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )

        self.is_resume_from_interrupt = True
        return train_output
