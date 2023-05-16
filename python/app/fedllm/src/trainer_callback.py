from pathlib import Path

from peft import PeftModel
import torch
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        if state.is_world_process_zero or (state.is_local_process_zero and args.save_on_each_node):
            # see https://github.com/huggingface/peft/issues/96#issuecomment-1460080427
            checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            model = kwargs.get("model", None)

            if isinstance(model, PeftModel):
                # when using DeepSpeed Zero 3, model weights need to be converted.
                # conversion is done by Trainer, we need to load the saved weights manually
                checkpoint = torch.load(str(checkpoint_dir / "pytorch_model.bin"), map_location="cpu")

                peft_model_path = checkpoint_dir / "adapter_model"
                model.save_pretrained(str(peft_model_path), state_dict=checkpoint)

        return control
