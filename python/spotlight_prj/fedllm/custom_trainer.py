"""Custom trainer that properly handles non-PEFT models for full fine-tuning.

This fixes a bug in FedML where set_model_params tries to call set_peft_model_state_dict
even when peft_type="none" is configured, causing AttributeError for non-PEFT models.

This version also integrates GRPO training for GSM8K dataset.
"""



import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
import torch
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from accelerate.utils import broadcast_object_list
from datasets import load_dataset
from fedml.train.llm.modeling_utils import to_device
from fedml.train.llm.distributed import barrier
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import prepare_deepspeed

# Fallback stub if prepare_fsdp is unavailable in current TRL version
try:
    from trl.trainer.utils import prepare_fsdp  # type: ignore
except ImportError:  # pragma: no cover
    def prepare_fsdp(model, accelerator):
        """Minimal FSDP prep fallback – just use accelerator.prepare_model."""
        return accelerator.prepare_model(model, evaluation_mode=True)

# Optional: stub SyncRefModelCallback if not provided upstream
try:
    from trl.trainer.callbacks import SyncRefModelCallback  # hypothetical future addition
except Exception:
    from transformers import TrainerCallback
    class SyncRefModelCallback(TrainerCallback):
        """Fallback no-op callback used when TRL doesn't ship one.
        Simply keeps reference model on correct device and in eval mode.
        """
        def __init__(self, ref_model=None, accelerator=None):
            self.ref_model = ref_model
            self.accelerator = accelerator
        def on_train_begin(self, args, state, control, **kwargs):
            if self.ref_model is not None and self.accelerator is not None:
                self.ref_model.to(self.accelerator.device)
                self.ref_model.eval()
from fedml.ml.aggregator.agg_operator import FedMLAggOperator

from run_fedllm import LLMTrainer, LLMAggregator, save_checkpoint, load_checkpoint
from src.peft_utils import set_peft_model_state_dict
from src.modeling_utils import load_state_dict
import time, logging
import threading
import subprocess  # for launching validation after checkpoints
import shutil  # for deleting old checkpoints

from fractions import Fraction

# New import for TrainerCallback
from transformers import TrainerCallback, AutoConfig, AutoModelForCausalLM
import transformers

import wandb
import json

import warnings
warnings.filterwarnings("ignore")

#import gc


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    """
    Disable dropout by setting all torch.nn.Dropout modules to eval mode and
    zero probability.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            module.eval()


class TimedGRPOTrainer(GRPOTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ref_model is not None:
            # Load any model you like as the reference baseline
            self.ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-GPTQ-Int8")
            self.ref_model.eval()
            disable_dropout_in_model(self.ref_model)
            # Move reference model to the same device as the policy so that
            # inputs and weights reside on a single device (avoids CPU↔GPU mismatch).
            # `Trainer` already initialises an `accelerator` attribute so we can
            # rely on `self.accelerator.device` to pick the correct target.
            self.ref_model.to(self.accelerator.device)
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        
        # Keep the commented line for quick CPU off-loading during debugging
        # self.ref_model.to('cpu')
    

    def _record_step_stats(self, stats):
        # -------------------------------------------------------------
        # Measure *inter-step* wall-clock time: difference between the start
        # of this stats call and the previous.  This captures the full time
        # spent in the GRPO optimisation step (generation + backward pass,
        # etc.) rather than just the duration of this method.
        # -------------------------------------------------------------
        t_now = time.perf_counter()
        step_elapsed = None
        if hasattr(self, "_prev_step_t"):
            step_elapsed = t_now - self._prev_step_t
        self._prev_step_t = t_now  # update for next call

        # Call parent implementation *after* timing start so that we include
        # all work done before stats are returned.
        super()._record_step_stats(stats)

        # -------------------------------------------------------------
        # Compute additional metrics
        # -------------------------------------------------------------
        stats["kl_divergence"] = stats["kl"].mean().item()
        if step_elapsed is not None:
            stats["grpo_step_time"] = step_elapsed  # seconds

        # NEW: forward stats to Trainer's logging system so that callbacks
        # like GRPOMetricsCallback can record them via the TrainingMetricsLogger.
        # This ensures that after every GRPO step the metrics are properly
        # captured by the custom logger.
        self.log(stats)
    
    # Override GRPOTrainer internals to measure generation latency per roll-out batch
    # NOTE: Upstream `GRPOTrainer` uses `_generate_and_score_completions` (not
    # `_make_experience`).  The original override therefore never executed.
    # We rename the method accordingly so that it is invoked during training.

    def _generate_and_score_completions(self, *args, **kwargs):
        
        t0 = time.perf_counter()
        # Call upstream implementation
        result = super()._generate_and_score_completions(*args, **kwargs)
        # ------------------------------------------------------------------
        # Compute and log average completion time per generation
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - t0  # total time for this roll-out batch
        num_gens = max(1, getattr(self.args, "num_generations", 1))
        self.avg_completion_time = elapsed / num_gens

        # Log the metric so that it is captured by both Accelerate and
        # the TrainingMetricsLogger (via GRPOMetricsCallback).
        self.accelerator.log({"avg_completion_time": self.avg_completion_time}, step=self.state.global_step)
        self.log({"avg_completion_time": self.avg_completion_time})

        if self.state.global_step % 10 == 0:
            torch.cuda.empty_cache()

        return result


class FullModelLLMTrainer(LLMTrainer):
    """Custom trainer that properly handles both PEFT and non-PEFT models with GRPO training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # GSM8K specific regex patterns
        # Regex for GSM8K dataset format (####)
        self.DATASET_ANS = re.compile(r"####\s*([-+]?\d+\.?\d*)")
        # Regex for model completion format (\boxed{})
        self.MODEL_ANS = re.compile(r"\\boxed\{([^}]*)\}")

        self.BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")  # capture content inside \boxed{…}

        # ------------------------------------------------------------------
        # Configuration: enable or disable per-round checkpoints
        # ------------------------------------------------------------------

        # Default: omit per-round checkpoints unless user explicitly enables
        # them via the FedML YAML (enable_round_checkpoints: true)
        self._enable_round_ckpt = getattr(self.args, "enable_round_checkpoints", False)

        self.exact_match_reward = 2.0
        self.numeric_equivalence_reward=1.5
        self.incorrect_answer_reward=0.0

        # Instantiate the training metrics logger and keep as an attribute so
        # it can be accessed by callbacks.
        self.logger = TrainingMetricsLogger(
            log_dir=os.path.join(self.args.output_dir, "wandb_logs"),
            run_name=f"fl-client{getattr(self.args, 'rank', 'unknown')}_run{getattr(self.args, 'run_id', os.getenv('FEDML_CURRENT_RUN_ID', '0'))}",
            enable_wandb=True,
            wandb_project="fedllm-grpo-training",
            args=self.args,
        )

    
    def to_number(self, text: str) -> Optional[float]:
        """Convert string to float if possible, handling simple fractions."""
        text = text.replace(",", "").strip()
        # Fractions like 3/4
        if "/" in text:
            try:
                return float(Fraction(text))
            except (ValueError, ZeroDivisionError):
                pass
        try:
            return float(text)
        except ValueError:
            return None


    def extract_boxed(self, text: str) -> str:
        """Return first \\boxed{...} contents; '' if none."""
        m = self.BOXED_RE.search(text)
        return m.group(1) if m else ""
    
    def reward_fn(self, completions, answer, **_):
        """Reward function for GSM8K that checks if the predicted answer matches the true answer."""
        out = []
        for c, ans in zip(completions, answer):
            # Extract from dataset answer (GSM8K format)
            tru = self.DATASET_ANS.search(ans)
            # Extract from model completion (boxed format, fallback to GSM8K format)
            pred = self.MODEL_ANS.search(c)
            if not pred:
                pred = self.DATASET_ANS.search(c)
            
            if pred and tru:
                pred_num = pred.group(1)
                tru_num = tru.group(1)
                if pred_num == tru_num:
                    out.append(self.exact_match_reward)
                else:
                    p_num, g_num = self.to_number(pred_num), self.to_number(tru_num)
                    if (p_num is not None and g_num is not None and abs(p_num - g_num) < 1e-4):
                        out.append(self.numeric_equivalence_reward)
                    else:
                        out.append(self.incorrect_answer_reward)
            else:
                out.append(self.incorrect_answer_reward)
        return out
    
    def train(self, train_data, device, args):
        """Override train to use GRPO training on GSM8K dataset."""
        self.log("Starting GRPO training on GSM8K")

        
        # Load GSM8K dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.rename_column("question", "prompt")
        
        # Get GRPO-specific settings from FedML config or use defaults
        grpo_max_steps = getattr(args, 'grpo_max_steps', -1)  # -1 means use epochs
        grpo_num_epochs = getattr(args, 'grpo_num_epochs', 3)
        grpo_batch_size = getattr(args, 'grpo_batch_size', 32)
        
        # Calculate effective batch size for GRPO constraint
        # effective_batch_size = num_gpus * per_device_batch_size * gradient_accumulation_steps
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 2)
        effective_batch_size = 1 * grpo_batch_size * gradient_accumulation_steps
        
        # Num generations must evenly divide the effective batch size
        # For testing with small batch sizes, use a smaller num_generations
        if effective_batch_size >= 64:
            num_generations = 64
        elif effective_batch_size >= 32:
            num_generations = 32
        elif effective_batch_size >= 16:
            num_generations = 16
        elif effective_batch_size >= 8:
            num_generations = 8
        else:
            num_generations = 2
        
        num_generations = 2
        
        # For testing, we can use a very small number of steps
        if grpo_max_steps > 0:
            self.log(f"GRPO training for {grpo_max_steps} steps (test mode)")
        else:
            self.log(f"GRPO training for {grpo_num_epochs} epochs")
        
        self.log(f"Using num_generations={num_generations} with effective batch size={effective_batch_size}")
        
        # **FIX: Load fresh model and tokenizer for GRPO to avoid FedML state corruption**
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # ↓↓↓  off-load the FedML copy BEFORE allocating fresh_model
        self.model.to("cpu")
        torch.cuda.empty_cache()       # actually releases the VRAM
        
        # Get model name from model_args
        model_name = self.model_args.model_name_or_path
        self.log(f"Loading fresh model and tokenizer: {model_name}")
        
        # Load fresh model and tokenizer with numerical stability
        try:
            # Try bfloat16 first if requested
            if args.bf16:
                fresh_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.bfloat16,
                    use_cache=False,
                    trust_remote_code=True
                )
            else:
                fresh_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,  # Use float32 for better stability
                    use_cache=False,
                    trust_remote_code=True
                )
        except Exception as e:
            self.log(f"Failed to load with requested precision, falling back to float32: {e}")
            fresh_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,  # Fallback to float32
                use_cache=False,
                trust_remote_code=True
            )
        fresh_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        fresh_tokenizer.pad_token = fresh_tokenizer.eos_token
        
        # Copy current model state to fresh model (to preserve any training from previous rounds)
        if self.round_idx > 0:
            self.log("Copying trained weights to fresh model")
            # Get the current model state dict (handling potential PEFT wrapping)
            if isinstance(self.model, PeftModel):
                current_state = self.model.base_model.state_dict()
            else:
                current_state = self.model.state_dict()
            
            # Load into fresh model
            incompatible = fresh_model.load_state_dict(current_state, strict=True)
            # Log any keys that failed to load for easier debugging
            self.log(
                f"missing keys: {incompatible.missing_keys}, unexpected keys: {incompatible.unexpected_keys}"
            )
        
        # Move fresh model to correct device
        fresh_model.to(device)
        
        # **FIX: Additional model preparation for numerical stability**
        fresh_model.eval()  # Set to eval mode initially
        
        # Ensure model is in proper state for training
        for param in fresh_model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.log("WARNING: Found NaN/Inf in model parameters, reinitializing")
                param.data.normal_(0, 0.02)  # Reinitialize problematic parameters
        
        fresh_model.train()  # Set back to train mode
        
        self.log(f"Fresh model loaded: dtype={fresh_model.dtype}, device={next(fresh_model.parameters()).device}")
        self.log(f"Tokenizer vocab size: {len(fresh_tokenizer)}, pad_token_id: {fresh_tokenizer.pad_token_id}")
        
        # Configure GRPO training
        # Match precision to model dtype
        use_bf16 = fresh_model.dtype == torch.bfloat16
        cfg = GRPOConfig(
            output_dir=str(self.checkpoint_dir / "grpo"),
            per_device_train_batch_size=grpo_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_completion_length=512,
            num_generations=num_generations,  # Adjusted based on effective batch size
            num_train_epochs=grpo_num_epochs if grpo_max_steps <= 0 else 1,  # Use 1 epoch if max_steps is set
            max_steps=grpo_max_steps if grpo_max_steps > 0 else -1,  # Override epochs with max_steps
            learning_rate=5e-6,
            bf16=use_bf16,  # Match model precision
            fp16=not use_bf16,  # Use fp16 if not bf16
            gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
            #logging_steps=5 if grpo_max_steps > 0 and grpo_max_steps < 50 else 25,  # More frequent logging for short runs
            logging_steps=1,
            log_completions=False,
            save_steps=grpo_max_steps if grpo_max_steps > 0 else 500,  # Save at the end if using max_steps
            # Add seed for reproducibility in federated setting
            seed=int(time.perf_counter_ns() % (2**32)),
            #report_to="wandb",
            scale_rewards=False,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            epsilon=0.2,
            beta=0.1,
            optim="sgd",
        )
        
        self.log(f"GRPO Config - bf16: {use_bf16}, fp16: {not use_bf16}, batch_size: {grpo_batch_size}")
        self.log(f"GRPO Config - max_completion_length: 1024, num_generations: {num_generations}")
        
        # Create GRPO trainer with fresh model and tokenizer
        grpo_trainer = TimedGRPOTrainer(
            model=fresh_model,  # Use fresh model
            args=cfg,
            train_dataset=ds.shuffle(seed=cfg.seed),
            processing_class=fresh_tokenizer,  # Use fresh tokenizer
            reward_funcs=self.reward_fn,
        )
        
        # **FIX: Set generation parameters for numerical stability**
        grpo_trainer.generation_kwargs = {
            "do_sample": True,
            "pad_token_id": fresh_tokenizer.eos_token_id,
            "eos_token_id": fresh_tokenizer.eos_token_id,
            "bos_token_id": fresh_tokenizer.bos_token_id,
            "max_new_tokens": 512,
            "length_penalty": 1.0,      # Neutral length penalty
        }
        
        # Attach our logging callback so that metrics are recorded every step.
        grpo_trainer.add_callback(GRPOMetricsCallback(self.logger))

        self.log(f"Set generation parameters: {grpo_trainer.generation_kwargs}")
        
        # Run GRPO training
        grpo_trainer.train()

        

        # **Copy trained weights back to FedML's model**
        self.log("Copying GRPO-trained weights back to FedML model")
        trained_state = fresh_model.state_dict()
        
        # Load into FedML model (handling potential PEFT wrapping)
        if isinstance(self.model, PeftModel):
            self.model.base_model.load_state_dict(trained_state, strict=False)
        else:
            self.model.load_state_dict(trained_state, strict=False)
        self.model.to("cpu")
        #del trained_state
        
        # Optionally save a pre-aggregation checkpoint for this round

        self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_before_agg"
        self.log(f"[round-ckpt] Saving GRPO-trained model to \"{self.latest_checkpoint_dir}\"")

        save_checkpoint(
            self.model,
            self.latest_checkpoint_dir,
            is_saving_process=self.training_args.should_save,
            synchronize=True
        )

        # After saving the current round checkpoint, clean up older round_* checkpoints
        if self.training_args.should_save:
            self._cleanup_old_round_checkpoints()
        """
        grpo_trainer.accelerator.end_training()
        grpo_trainer.accelerator.free_memory()
        grpo_trainer.model = None
        gc.collect()
        torch.cuda.empty_cache()
        """
        
        # Clean up fresh model to free memory
        del fresh_model
        del fresh_tokenizer
        del grpo_trainer.optimizer
        del grpo_trainer.lr_scheduler
        del grpo_trainer
        self.model.to("cpu")
        torch.cuda.empty_cache()
        
        self.log("GRPO training finished")
    
    def on_after_local_training(self, train_data, device, args):
        """Override to skip the parent's checkpoint saving since we handle it in train()."""
        self.log("Skipping parent's on_after_local_training (already saved in train method)")
        # We already saved the checkpoint in the train() method, so we don't need to do anything here
        # This prevents the AttributeError from trying to save the trainer's optimizer state
        return None
    
    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        t0 = time.perf_counter()

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        # Check if model is a PEFT model
        if isinstance(self.model, PeftModel):
            set_peft_model_state_dict(self.model, model_parameters)
        else:
            # For non-PEFT models, use regular load_state_dict
            load_state_dict(self.model, model_parameters, strict=False)
        barrier()



        # save aggregated model checkpoint
        self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_after_agg"
        self.log(f"saving aggregated model to \"{self.latest_checkpoint_dir}\"")
        save_checkpoint(
            self.model,
            self.latest_checkpoint_dir,
            is_saving_process=self.training_args.should_save,
            state_dict=model_parameters,
            synchronize=True
        )
        
        elapsed = time.perf_counter() - t0
        self.log(f"set_model_params (client) took {elapsed:.3f}s")

        self.log("finished")
    
    # Explicitly define sync_process_group to ensure FedML recognizes it
    def sync_process_group(
            self,
            round_idx: Optional[int] = None,
            model_params: Optional[Any] = None,
            client_index: Optional[int] = None,
            from_process: int = 0
    ) -> None:
        self.log("start")

        if round_idx is None:
            round_idx = self.round_idx

        model_params = to_device(model_params, "cpu")   # ensure params live on CPU

        dtypes = set(t.dtype for t in model_params.values())
        print(f"model_params dtypes: {dtypes}")  # Should print torch.float32 if FP32

        broadcast_object_list([round_idx, model_params, client_index], from_process=from_process)

        self.log("finished")

    def await_sync_process_group(self, from_process: int = 0) -> list:
        self.log("start")

        # ---------------------- Timing start ----------------------
        t0 = time.perf_counter()
        outputs = broadcast_object_list([None, None, None], from_process=from_process)
        download_elapsed = time.perf_counter() - t0

        # ---------------------- WandB log ------------------------
        if getattr(self, "logger", None) and self.logger.enable_wandb and self.logger.wandb_run:
            # Step keyed by federated round so uploads and downloads align.
            self.logger.wandb_run.log({
                "performance/model_download_time": download_elapsed
            }, step=self.round_idx)

            # Store for optional moving-average statistics.
            self.logger.accumulated_metrics.setdefault("model_download_times", []).append(download_elapsed)

        self.log(f"model download took {download_elapsed:.3f}s")
        self.log("finished")
        return outputs

    def _cleanup_old_round_checkpoints(self, keep_last: int = 1):
        """Delete old round_* checkpoints but keep the most recent `keep_last`.

        Wall-clock checkpoints (wallclock_*) are never removed.
        """
        pattern = re.compile(r"round_(\d+)_(before|after)_agg")
        # Collect candidate directories and their round numbers
        ckpts = []
        for d in self.checkpoint_dir.iterdir():
            m = pattern.fullmatch(d.name)
            if m and d != self.latest_checkpoint_dir:
                ckpts.append((int(m.group(1)), d))

        # Sort by round number so oldest come first
        ckpts.sort(key=lambda x: x[0])

        # Remove all but the newest `keep_last` checkpoints
        for _, d in ckpts[:-keep_last]:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                self.log(f"[WARN] Failed to delete old checkpoint {d}: {e}")


class FullModelLLMAggregator(LLMAggregator):
    """Custom aggregator that properly handles both PEFT and non-PEFT models."""
    
    # ------------------------------------------------------------------
    # Periodic checkpointing setup
    # ------------------------------------------------------------------

    def __init__(self, *args, **kwargs):
        """Extend parent init and start a background thread that creates a
        checkpoint every ``server_checkpoint_interval_minutes`` (default 30).

        Notes
        -----
        * Only the main process (``self.is_main_process()``) actually writes the
          checkpoint to avoid race conditions.
        * Checkpoints are written under
          ``{self.checkpoint_dir}/wallclock_{unix_ts}`` so they will not
          collide with the per-round checkpoints that already exist.
        """
        super().__init__(*args, **kwargs)

        # -------------------------------------------------------------
        # WandB logger for aggregator-level (server) statistics – initialize
        # EARLY so that it exists even when periodic checkpointing is disabled.
        # -------------------------------------------------------------
        self.logger = TrainingMetricsLogger(
            log_dir=os.path.join(self.args.output_dir, "wandb_logs"),
            run_name=f"fl-server_run{getattr(self.args, 'run_id', os.getenv('FEDML_CURRENT_RUN_ID', '0'))}",
            enable_wandb=True,
            wandb_project="fedllm-grpo-training",
            args=self.args,
        )
        self.model_broadcasts = 0

        # Determine interval (seconds)
        interval_min = getattr(self.args, "server_checkpoint_interval_minutes", 30)
        if interval_min <= 0:
            # Disable if user passes 0 or negative value
            self._checkpoint_interval = None
            return

        self._checkpoint_interval = interval_min * 60

        # Background thread is only needed on the main process
        if self.is_main_process():
            self._stop_checkpoint_evt = threading.Event()
            self._checkpoint_thread = threading.Thread(
                target=self._periodic_checkpoint_loop,
                name="periodic_ckpt_thread",
                daemon=True,
            )
            self._checkpoint_thread.start()

        # Whether to save per-round checkpoints (default False)
        self._enable_round_ckpt = getattr(self.args, "enable_round_checkpoints", False)

        # ------------------ Nesterov Momentum Setup (NEW) ------------------
        # Learning rate for the server optimizer (default 1.0 so the server fully
        # applies the aggregated update when momentum=0)
        self._server_lr = getattr(self.args, "server_lr", 1.0)
        # Momentum coefficient. Typical values are 0.9 or 0.99
        self._momentum = getattr(self.args, "server_momentum", 0.9)
        # Enable / disable Nesterov variant (default=True)
        self._nesterov = getattr(self.args, "server_nesterov", True)
        # Momentum buffer for each parameter
        self._velocity: OrderedDict = OrderedDict()
        # -------------------------------------------------------------------

        # ----- WandB server-side logger (NEW) -----
        # Create a standalone TrainingMetricsLogger so that aggregator-level
        # system statistics (e.g. active workers, model broadcasts) are also
        # recorded in the same WandB project as the clients.
        if not hasattr(self, "logger"):
            self.logger = TrainingMetricsLogger(
                log_dir=os.path.join(self.args.output_dir, "wandb_logs"),
                run_name=f"fl-server_run{getattr(self.args, 'run_id', os.getenv('FEDML_CURRENT_RUN_ID', '0'))}",
                enable_wandb=True,
                wandb_project="fedllm-grpo-training",
                args=self.args,
            )
            # Counter for how many times the global model has been broadcast to
            # clients – useful for monitoring server throughput.
            self.model_broadcasts = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_old_round_checkpoints(self, keep_last: int = 1):
        """Delete old round_* checkpoints but keep the most recent `keep_last`.

        Wall-clock checkpoints (wallclock_*) are never removed.
        """
        pattern = re.compile(r"round_(\d+)_(before|after)_agg")
        # Collect candidate directories and their round numbers
        ckpts = []
        for d in self.checkpoint_dir.iterdir():
            m = pattern.fullmatch(d.name)
            if m and d != self.latest_checkpoint_dir:
                ckpts.append((int(m.group(1)), d))

        # Sort by round number so oldest come first
        ckpts.sort(key=lambda x: x[0])

        # Remove all but the newest `keep_last` checkpoints
        for _, d in ckpts[:-keep_last]:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                self.log(f"[WARN] Failed to delete old checkpoint {d}: {e}")

    def _periodic_checkpoint_loop(self):
        """Loop that sleeps ``_checkpoint_interval`` seconds then writes a
        checkpoint until ``_stop_checkpoint_evt`` is set (i.e., program exit).
        """
        while not self._stop_checkpoint_evt.wait(self._checkpoint_interval):
            try:
                ts = int(time.time())
                ckpt_dir = self.checkpoint_dir / f"wallclock_{ts}"
                self.log(f"Periodic checkpoint → {ckpt_dir}")
                # Always save checkpoints in the standard HuggingFace format so that
                # the resulting directory can be loaded with `from_pretrained`.
                # For `PeftModel` this will also persist the adapter weights.
                # Only the main process writes the checkpoint to avoid race conditions
                # (the background thread is spawned exclusively on the main process).
                if self.training_args.should_save:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        # Try the native HuggingFace save.
                        # For `PeftModel` this will also persist the adapter weights.
                        self.model.save_pretrained(str(ckpt_dir), state_dict=self.model.state_dict())
                    except AttributeError:
                        # Fallback to the generic helper if the model doesn't implement
                        # `save_pretrained` (unlikely for LLMs but safe-guard regardless).
                        save_checkpoint(
                            self.model,
                            checkpoint_dir=ckpt_dir,
                            is_saving_process=True,
                            synchronize=False,
                        )
                    # ---------------- New behaviour ----------------
                    # After successfully writing the checkpoint, prune older
                    # wallclock_* checkpoints so that only the latest six are
                    # kept on disk.
                    self._cleanup_old_wallclock_checkpoints()
                    # Run validation on the newly saved checkpoint
                    try:
                        script_path = Path(__file__).parent / "validation.py"
                        log_path = Path(self.args.output_dir) / "validation.log"
                        with open(log_path, "a") as lf:
                            subprocess.Popen(
                                [sys.executable, str(script_path), "--model", str(ckpt_dir)],
                                stdout=lf,
                                stderr=subprocess.STDOUT,
                                close_fds=True,
                            )
                    except Exception as e:
                        self.log(f"[WARN] Failed to launch validation: {e}")
            except Exception as e:
                # Log and continue – do not crash training due to checkpoint failure
                self.log(f"[WARN] Periodic checkpoint failed: {e}")

    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        t0 = time.perf_counter()

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        # Check if model is a PEFT model
        if isinstance(self.model, PeftModel):
            set_peft_model_state_dict(self.model, model_parameters)
        else:
            # For non-PEFT models, use regular load_state_dict
            load_state_dict(self.model, model_parameters, strict=False)
        barrier()

        # save aggregated model checkpoint
        self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_after_agg"
        self.log(f"saving aggregated model to \"{self.latest_checkpoint_dir}\"")
        save_checkpoint(
            self.model,
            self.latest_checkpoint_dir,
            is_saving_process=self.training_args.should_save,
            state_dict=model_parameters,
            synchronize=True
        )

        # Clean up old round checkpoints on the server as well
        if self.training_args.should_save:
            self._cleanup_old_round_checkpoints()
        
        elapsed = time.perf_counter() - t0
        self.log(f"set_model_params (server) took {elapsed:.3f}s")

        # -------------------------------------------------------------
        # NEW: push aggregator-level system statistics to WandB
        # -------------------------------------------------------------
        self.model_broadcasts += 1
        self.logger.log_server_statistics(
            stats={
                "server_statistics": {
                    "active_workers": getattr(self.args, "client_num_in_total", 0),
                    "model_subscribers": [],
                    "service_status": {
                        "current_model_version": self.round_idx,
                        "buffer_statistics": {},
                    },
                },
                "current_pipeline_depth": 0,  # placeholder – update if pipeline depth is tracked elsewhere
                "model_broadcasts": self.model_broadcasts,
            },
            global_step=self.round_idx,
        )

        self.log("finished")

    def _cleanup_old_wallclock_checkpoints(self, keep_last: int = 6):
        """Delete old wallclock_* checkpoints but keep the most recent ``keep_last``.

        This complements the round-based checkpoint cleanup by pruning time-based
        checkpoints created by the periodic background thread.  The newest
        ``keep_last`` checkpoints are retained; older ones are removed to avoid
        unbounded disk usage on long-running servers.
        """
        pattern = re.compile(r"wallclock_(\d+)$")
        valid_ckpts = []  # (timestamp, Path)
        invalid_ckpts = []  # Path(s) that lack model files

        # Determine candidate checkpoints and group by validity
        for d in self.checkpoint_dir.iterdir():
            m = pattern.fullmatch(d.name)
            if not m:
                continue  # skip non-wallclock dirs

            # Heuristic: consider checkpoint *valid* if it contains at least one
            # model weight file produced by ``save_pretrained`` or our fallback
            # helper (i.e. *.bin or *.safetensors).  This covers both HF and PEFT.
            has_model_file = any(d.glob("*.bin")) or any(d.glob("*.safetensors")) or any(d.glob("*.pt"))

            if has_model_file:
                valid_ckpts.append((int(m.group(1)), d))
            else:
                invalid_ckpts.append(d)

        # Remove *all* invalid checkpoints immediately as they are unusable
        for d in invalid_ckpts:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                self.log(f"[WARN] Failed to delete incomplete wallclock checkpoint {d}: {e}")

        # Sort valid checkpoints chronologically (oldest first)
        valid_ckpts.sort(key=lambda x: x[0])

        # Keep only the most recent ``keep_last`` valid checkpoints
        for _, d in valid_ckpts[:-keep_last]:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                self.log(f"[WARN] Failed to delete old wallclock checkpoint {d}: {e}")


class TrainingMetricsLogger:
    """Comprehensive logging for GRPO training with WandB support"""

    def __init__(
        self,
        log_dir: str,
        run_name: Optional[str] = None,
        enable_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[dict] = None,
        args: Optional[Any] = None,
    ):
        """Parameters
        ----------
        log_dir : str
            Directory where auxiliary JSON / txt logs will be written.
        run_name : str, optional
            Human-readable name that will appear in the WandB UI.
        enable_wandb : bool, default False
            If ``True`` a WandB run is initialised, otherwise the logger will
            operate in offline mode and simply discard `.log*()` calls.
        wandb_project, wandb_entity, wandb_config : Optional[str | dict]
            Passed through to :pyfunc:`wandb.init` unchanged.
        args : Any, optional
            (FedML) *args* namespace used throughout the project.  We only
            use it to derive a *unique* WandB run *id* so that the server and
            every client write to **separate** runs instead of clobbering one
            another.
        """

        self.log_dir = log_dir
        self.run_name = run_name or f"grpo_training_{int(time.time())}"
        self.enable_wandb = enable_wandb
        self.args = args  # may be ``None`` for unit tests / offline runs

        # ------------------------------------------------------------------
        # WandB setup – ensure that each process (server / client-rank-N) gets
        # its *own* run.  Re-using the same run *id* from multiple processes
        # causes metrics to silently overwrite each other and leads to exactly
        # the "not everything we log shows up" behaviour that we observed on
        # the dashboard.
        # ------------------------------------------------------------------
        self.wandb_run = None
        if self.enable_wandb:
            wandb_kwargs = {
                "project": wandb_project or "grpo-training",
                "entity": wandb_entity,
                "name": self.run_name,
                "config": wandb_config or {},
                "reinit": True,
            }

            # Use a *group* so that the server run and all client runs are
            # nicely collated in the WandB UI, while still receiving unique
            # run IDs.
            if args is not None and hasattr(args, "run_id"):
                wandb_kwargs["group"] = str(args.run_id)

                # Derive a UNIQUE id: "<run_id>-server"  or  "<run_id>-client<rank>"
                role_suffix = (
                    "-server"
                    if getattr(args, "role", "server") == "server"
                    else f"-client{getattr(args, 'rank', '0')}"
                )
                wandb_kwargs["id"] = f"{args.run_id}{role_suffix}"

            # Remove None entries so wandb.init does not complain.
            wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}

            self.wandb_run = wandb.init(**wandb_kwargs)
            print(
                f"[WandB] Logging initialised → "
                f"project={wandb_kwargs.get('project')}, run_name={self.run_name}"
            )

        # Metrics tracking
        self.step_count = 0
        self.training_start_time = time.time()
        self.last_log_time = time.time()
        # Stores the most recent average completion time reported by the trainer.
        # Initialised here so that attribute always exists and we avoid AttributeError
        # if the metric is accessed before the first value is logged.
        self.avg_completion_time: Optional[float] = None

        # Accumulated metrics for averaging
        self.accumulated_metrics = {
            'losses': [],
            'rewards': [],
            'kl_divergences': [],
            'policy_losses': [],
            'value_losses': [],
            'advantages': [],
            'rollout_lengths': [],
            'completion_times': [],
            'step_times': [],
        }

    def log_training_step(self, step_id: str, train_result: dict, global_step: int):
        """Log metrics for a single training step"""
        
        # Prepare metrics dict for wandb
        wandb_metrics = {}

        # Core training metrics
        if 'loss' in train_result:
            wandb_metrics['training/loss'] = train_result['loss']
            self.accumulated_metrics['losses'].append(train_result['loss'])

        if 'avg_reward' in train_result:
            wandb_metrics['training/avg_reward'] = train_result['avg_reward']
            self.accumulated_metrics['rewards'].append(train_result['avg_reward'])

        # Advanced GRPO metrics
        if 'kl_divergence' in train_result:
            wandb_metrics['training/kl_divergence'] = train_result['kl_divergence']
            self.accumulated_metrics['kl_divergences'].append(train_result['kl_divergence'])

        if 'policy_loss' in train_result:
            wandb_metrics['training/policy_loss'] = train_result['policy_loss']
            self.accumulated_metrics['policy_losses'].append(train_result['policy_loss'])

        if 'value_loss' in train_result:
            wandb_metrics['training/value_loss'] = train_result['value_loss']
            self.accumulated_metrics['value_losses'].append(train_result['value_loss'])

        if 'advantage_mean' in train_result:
            wandb_metrics['training/advantage_mean'] = train_result['advantage_mean']
            self.accumulated_metrics['advantages'].append(train_result['advantage_mean'])

        # Rollout statistics
        if 'rollout_count' in train_result:
            wandb_metrics['rollouts/count_per_step'] = train_result['rollout_count']

        if 'avg_rollout_length' in train_result:
            wandb_metrics['rollouts/avg_length'] = train_result['avg_rollout_length']
            self.accumulated_metrics['rollout_lengths'].append(train_result['avg_rollout_length'])

        # Average completion time (per generation)
        # Update the cached value if the trainer provided a fresh measurement.
        if 'avg_completion_time' in train_result:
            self.avg_completion_time = train_result['avg_completion_time']

        if self.avg_completion_time is not None:
            wandb_metrics['performance/avg_completion_time'] = self.avg_completion_time
            self.accumulated_metrics['completion_times'].append(self.avg_completion_time)

        if 'rollout_time' in train_result:
            wandb_metrics['performance/rollout_time'] = train_result['rollout_time']

        if 'training_time' in train_result:
            wandb_metrics['performance/training_step_time'] = train_result['training_time']

        # Weight update timing metrics
        if 'weight_update_time' in train_result:
            wandb_metrics['performance/weight_update_time'] = train_result['weight_update_time']

        if 'backward_time' in train_result:
            wandb_metrics['performance/backward_pass_time'] = train_result['backward_time']

        if 'optimizer_time' in train_result:
            wandb_metrics['performance/optimizer_step_time'] = train_result['optimizer_time']

        if 'wait_time' in train_result:
            wandb_metrics['performance/batch_wait_time'] = train_result['wait_time']

        # Gradient metrics
        if 'grad_norm' in train_result:
            wandb_metrics['training/grad_norm'] = train_result['grad_norm']

        # Learning rate
        if 'learning_rate' in train_result:
            wandb_metrics['training/learning_rate'] = train_result['learning_rate']

        # GRPO step time
        if 'grpo_step_time' in train_result:
            wandb_metrics['performance/grpo_step_time'] = train_result['grpo_step_time']
            self.accumulated_metrics['step_times'].append(train_result['grpo_step_time'])

        # Log to wandb
        if self.enable_wandb and self.wandb_run and wandb_metrics:
            # Replace the Trainer-provided ``global_step`` (which resets every
            # round) with an internal monotonically-increasing counter so
            # that WandB treats each update as a new step instead of
            # overwriting previous values.
            wandb_step = self.step_count  # 0-based running counter
            wandb_metrics['global_step'] = wandb_step
            self.wandb_run.log(wandb_metrics, step=wandb_step)

        # Advance our own monotonically-increasing counter by exactly one
        # because this method is invoked once per call to `Trainer.log`.
        self.step_count += 1

    def log_server_statistics(self, stats: dict, global_step: int):
        """Log server and system statistics"""
        wandb_metrics = {}

        if 'server_statistics' in stats:
            server_stats = stats['server_statistics']

            # Handle double nesting
            if 'server_statistics' in server_stats:
                server_stats = server_stats['server_statistics']

            # Active workers
            if 'active_workers' in server_stats:
                wandb_metrics['system/active_workers'] = server_stats['active_workers']

            # Model subscribers
            if 'model_subscribers' in server_stats:
                inference_workers = [w for w in server_stats['model_subscribers'] if 'trainer' not in w.lower()]
                wandb_metrics['system/inference_workers'] = len(inference_workers)
                wandb_metrics['system/total_subscribers'] = len(server_stats['model_subscribers'])

            # Service status
            if 'service_status' in server_stats:
                service_status = server_stats['service_status']

                # Buffer statistics
                if 'buffer_statistics' in service_status:
                    buffer_stats = service_status['buffer_statistics']

                    if 'pending_steps' in buffer_stats:
                        wandb_metrics['system/pending_steps'] = buffer_stats['pending_steps']

                    if 'ready_batches' in buffer_stats:
                        wandb_metrics['system/ready_batches'] = buffer_stats['ready_batches']

                    if 'total_rollouts_received' in buffer_stats:
                        wandb_metrics['system/total_rollouts_received'] = buffer_stats['total_rollouts_received']

                # Model version tracking
                if 'current_model_version' in service_status:
                    wandb_metrics['system/current_model_version'] = service_status['current_model_version']

        # Pipeline statistics
        if 'current_pipeline_depth' in stats:
            wandb_metrics['system/pipeline_depth'] = stats['current_pipeline_depth']

        if 'model_broadcasts' in stats:
            wandb_metrics['system/model_broadcasts'] = stats['model_broadcasts']

        # Log to wandb
        if self.enable_wandb and self.wandb_run and wandb_metrics:
            self.wandb_run.log(wandb_metrics, step=self.step_count)

    def log_performance_metrics(self, global_step: int, training_rate: Optional[float] = None):
        """Log performance and timing metrics"""
        wandb_metrics = {}
        
        current_time = time.time()
        elapsed_time = current_time - self.training_start_time

        # Training rate
        if training_rate is not None:
            wandb_metrics['performance/training_rate_steps_per_hour'] = training_rate

        # Overall training time
        wandb_metrics['performance/elapsed_time_hours'] = elapsed_time / 3600

        # Steps per second (recent)
        time_since_last_log = current_time - self.last_log_time
        if time_since_last_log > 0 and hasattr(self, 'last_step_count'):
            steps_since_last = global_step - self.last_step_count
            steps_per_second = steps_since_last / time_since_last_log
            wandb_metrics['performance/steps_per_second'] = steps_per_second

        # Log to wandb
        if self.enable_wandb and wandb_metrics:
            self.wandb_run.log(wandb_metrics, step=global_step)

        self.last_log_time = current_time
        self.last_step_count = global_step


    def get_moving_average(values, window):
        if len(values) == 0:
            return 0
        window = min(window, len(values))
        return sum(values[-window:]) / window

    def log_moving_averages(self, global_step: int, window_size: int = 100):
        """Log moving averages of key metrics"""
        wandb_metrics = {}

        # Moving averages
        if self.accumulated_metrics['losses']:
            avg_loss = self.get_moving_average(self.accumulated_metrics['losses'], window_size)
            wandb_metrics[f'moving_avg/loss_{window_size}'] = avg_loss

        if self.accumulated_metrics['rewards']:
            avg_reward = self.get_moving_average(self.accumulated_metrics['rewards'], window_size)
            wandb_metrics[f'moving_avg/reward_{window_size}'] = avg_reward

        if self.accumulated_metrics['kl_divergences']:
            avg_kl = self.get_moving_average(self.accumulated_metrics['kl_divergences'], window_size)
            wandb_metrics[f'moving_avg/kl_divergence_{window_size}'] = avg_kl

        if self.accumulated_metrics['rollout_lengths']:
            avg_length = self.get_moving_average(self.accumulated_metrics['rollout_lengths'], window_size)
            wandb_metrics[f'moving_avg/rollout_length_{window_size}'] = avg_length

        if self.accumulated_metrics['completion_times']:
            avg_ct = self.get_moving_average(self.accumulated_metrics['completion_times'], window_size)
            wandb_metrics[f'moving_avg/completion_time_{window_size}'] = avg_ct

        if self.accumulated_metrics['step_times']:
            avg_st = self.get_moving_average(self.accumulated_metrics['step_times'], window_size)
            wandb_metrics[f'moving_avg/step_time_{window_size}'] = avg_st

        # Log to wandb
        if self.enable_wandb and wandb_metrics:
            # Use our internal monotonically-increasing counter so that these
            # points are not overwritten when `global_step` resets each round.
            wandb_step = max(0, self.step_count - 1)
            self.wandb_run.log(wandb_metrics, step=wandb_step)

    def log_hyperparameters(self, hparams: dict):
        """Log hyperparameters"""
        # Convert all values to scalars for TensorBoard
        scalar_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                scalar_hparams[key] = value
            elif isinstance(value, (str,list)):
                # TensorBoard doesn't handle strings well, so we'll just log them as text
                continue
            else:
                scalar_hparams[key] = float(value) if value is not None else 0.0

        # Log to wandb (wandb handles different types better)
        if self.enable_wandb:
            # Update wandb config with hyperparameters
            self.wandb_run.config.update(hparams)

    def log_model_statistics(self, model, global_step: int):
        """Log model-specific statistics"""
        wandb_metrics = {}

        # Model parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        wandb_metrics['model/total_parameters'] = total_params
        wandb_metrics['model/trainable_parameters'] = trainable_params

        # Parameter norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > 0:
            wandb_metrics['model/gradient_norm'] = total_norm

        # Weight norms by layer (sample a few to avoid too many metrics)
        layer_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                # Only log first few layers to wandb to avoid clutter
                if layer_count < 10:
                    wandb_metrics[f'model_weights/{name}_norm'] = param.data.norm().item()
                layer_count += 1

        # Log to wandb
        if self.enable_wandb and wandb_metrics:
            self.wandb_run.log(wandb_metrics, step=global_step)

    def log_reward_distribution(self, rewards: list, global_step: int):
        """Log reward distribution"""
        if rewards:
            if self.enable_wandb:
                wandb_metrics = {
                    'rewards/min': min(rewards),
                    'rewards/max': max(rewards),
                    'rewards/std': torch.tensor(rewards).std().item(),
                    'rewards/mean': sum(rewards) / len(rewards)
                }
                # Create histogram for wandb
                wandb_metrics['rewards/histogram'] = wandb.Histogram(rewards)
                self.wandb_run.log(wandb_metrics, step=global_step)

    def save_training_config(self, config: dict):
        """Save training configuration to file"""
        config_path = os.path.join(self.log_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"Training configuration saved to: {config_path}")

    def close(self):
        """Close logging connections"""
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.finish()
            print("WandB logging closed")

# -------------------- New Callback --------------------
class GRPOMetricsCallback(TrainerCallback):
    """HuggingFace Trainer callback that forwards log events to our
    TrainingMetricsLogger instance so that each GRPO step is recorded."""

    def __init__(self, logger: "TrainingMetricsLogger"):
        super().__init__()
        self.logger = logger

    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Forward the metrics dictionary to the TrainingMetricsLogger. This
        # fires after every call to `Trainer.log`, i.e. after each GRPO step.
        if logs:
            # Use a generic step_id; users can differentiate by global_step.
            self.logger.log_training_step("grpo_step", logs, state.global_step)
