from pathlib import Path
import shutil

from peft import PeftModel
import torch
from transformers import Trainer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import WEIGHTS_NAME as HF_WEIGHTS_NAME

from .typing import PathType
from .utils import is_deepspeed_module


class HFTrainer(Trainer):
    def save_checkpoint(self, output_dir: PathType, overwrite_peft_checkpoint: bool = True) -> None:
        output_dir = Path(output_dir)
        checkpoint_dir = Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        model = self.model
        model_wrapped = self.model_wrapped if self.model_wrapped is not None else self.model

        self._save_checkpoint(model_wrapped, trial=None)

        with self.args.main_process_first(local=self.args.save_on_each_node):
            if self.args.should_save:
                if (
                        overwrite_peft_checkpoint and
                        is_deepspeed_zero3_enabled() and
                        is_deepspeed_module(model_wrapped) and
                        isinstance(model, PeftModel)
                ):
                    # As of transformers 4.30.2, manually calling Trainer._save_checkpoint
                    # leads to incomplete PEFT checkpoint. Thus, need to overwrite the checkpoint
                    checkpoint = torch.load(str(checkpoint_dir / HF_WEIGHTS_NAME), map_location="cpu")
                    model.save_pretrained(str(checkpoint_dir), state_dict=checkpoint)
                    del checkpoint

                output_dir.mkdir(parents=True, exist_ok=True)
                # TODO: dirs_exist_ok is for python 3.8+, support older python
                shutil.copytree(str(checkpoint_dir), str(output_dir), dirs_exist_ok=True)
