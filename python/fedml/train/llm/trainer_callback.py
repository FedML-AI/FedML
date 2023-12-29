from typing import Any, Optional

import os
from pathlib import Path

from transformers import TrainerCallback, TrainerState, TrainingArguments, TrainerControl
from transformers.integrations import rewrite_logs
from transformers.trainer_utils import IntervalStrategy, PREFIX_CHECKPOINT_DIR

from .integrations import is_fedml_available
from .typing import PathType


class ExtraSaveCallback(TrainerCallback):
    def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        extra_save_steps = set(getattr(args, "extra_save_steps", set()))

        if args.save_strategy != IntervalStrategy.NO and state.global_step in extra_save_steps:
            control.should_save = True

        return control


class FedMLCallback(TrainerCallback):
    def __init__(self):
        self._mlops = None

        has_fedml = is_fedml_available()
        if not has_fedml:
            raise RuntimeError("FedMLCallback requires fedml to be installed. Run `pip install fedml`.")
        else:
            from fedml import mlops

            self._mlops = mlops

    @property
    def run_id(self) -> Optional[str]:
        return os.getenv("FEDML_CURRENT_RUN_ID", None)

    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: Optional[Any] = None,
            **kwargs
    ):
        if bool(self.run_id) and state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._mlops.log_metric(
                {**logs, "train/global_step": state.global_step},
                step=state.global_step,
                customized_step_key="train/global_step"
            )

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ):
        if bool(self.run_id) and state.is_world_process_zero:
            self.log_model(
                name=f"model-{self.run_id}-{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
                checkpoint_dir=str(Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            )

    def log_model(self, name: str, checkpoint_dir: PathType) -> None:
        if bool(self.run_id):
            # upload model to FedML MLOps Platform
            artifact = self._mlops.Artifact(name=name, type=self._mlops.ARTIFACT_TYPE_NAME_MODEL)
            artifact.add_dir(str(checkpoint_dir))
            self._mlops.log_artifact(artifact)
