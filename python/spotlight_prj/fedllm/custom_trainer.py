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
from fedml.ml.aggregator.agg_operator import FedMLAggOperator

from run_fedllm import LLMTrainer, LLMAggregator, save_checkpoint, load_checkpoint
from src.peft_utils import set_peft_model_state_dict
from src.modeling_utils import load_state_dict
import time, logging
import threading

from fractions import Fraction

# New import for TrainerCallback
from transformers import TrainerCallback

import wandb
import json


class TimedGRPOTrainer(GRPOTrainer):
    def _record_step_stats(self, stats):
        # first let the parent push its metrics
        super()._record_step_stats(stats)

        # add / overwrite any extra metrics and push once more
        stats["kl_divergence"] = stats["kl"].mean().item()
        self.accelerator.log(stats, step=self.state.global_step)

        # NEW: forward stats to Trainer's logging system so that callbacks
        # like GRPOMetricsCallback can record them via the TrainingMetricsLogger.
        # This ensures that after every GRPO step the metrics are properly
        # captured by the custom logger.
        self.log(stats)
    
    def _make_experience(self, *args, **kwargs):
        
        t0 = time.perf_counter()
        result = super()._make_experience(*args, **kwargs)
        self.accelerator.log(f"roll-out batch {self.state.global_step} : "
                     f"{time.perf_counter() - t0:.3f}s")
        
        # `out["kl"]` is a 1-D tensor of per-token KL values
        kl_mean = result["kl"].mean().item()

        # push to the FedML / accelerate logger – it will end up in client?.log
        self.log({"kl_divergence": kl_mean})

        self.log(
            f"roll-out batch {self.state.global_step} "
            f"(elapsed {time.perf_counter() - t0:.3f}s, kl={kl_mean:.4f})"
        )
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
        
        # For testing, we can use a very small number of steps
        if grpo_max_steps > 0:
            self.log(f"GRPO training for {grpo_max_steps} steps (test mode)")
        else:
            self.log(f"GRPO training for {grpo_num_epochs} epochs")
        
        self.log(f"Using num_generations={num_generations} with effective batch size={effective_batch_size}")
        
        # **FIX: Load fresh model and tokenizer for GRPO to avoid FedML state corruption**
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
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
                    use_cache=False
                )
            else:
                fresh_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float32,  # Use float32 for better stability
                    use_cache=False
                )
        except Exception as e:
            self.log(f"Failed to load with requested precision, falling back to float32: {e}")
            fresh_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,  # Fallback to float32
                use_cache=False
            )
        fresh_tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            gradient_checkpointing=False,  # Keep consistent with config
            logging_steps=5 if grpo_max_steps > 0 and grpo_max_steps < 50 else 25,  # More frequent logging for short runs
            log_completions=True,
            save_steps=grpo_max_steps if grpo_max_steps > 0 else 500,  # Save at the end if using max_steps
            # Add seed for reproducibility in federated setting
            seed=42 + self.round_idx * 100 + args.rank,  # Different seed per round and client
            #report_to="wandb",
            scale_rewards=False,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            epsilon=0.2,
            beta=0.1,
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
        
        # Optionally save a pre-aggregation checkpoint for this round

        self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_before_agg"
        self.log(f"[round-ckpt] Saving GRPO-trained model to \"{self.latest_checkpoint_dir}\"")

        save_checkpoint(
            self.model,
            self.latest_checkpoint_dir,
            is_saving_process=self.training_args.should_save,
            synchronize=True
        )

        
        # Clean up fresh model to free memory
        del fresh_model
        del fresh_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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

        broadcast_object_list([round_idx, model_params, client_index], from_process=from_process)

        self.log("finished")

    def await_sync_process_group(self, from_process: int = 0) -> list:
        self.log("start")

        outputs = broadcast_object_list([None, None, None], from_process=from_process)

        self.log("finished")
        return outputs


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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        
        elapsed = time.perf_counter() - t0
        self.log(f"set_model_params (server) took {elapsed:.3f}s")


        self.log("finished")

    """
    def aggregate(self, raw_client_model_list):
        Aggregate client models with Nesterov momentum.

        Steps
        -----
        1. Compute the FedAvg-style weighted average of client models (same as the
           default FedML behaviour).
        2. Treat the *difference* between the current global model and the
           aggregated model as the (negative) gradient.
        3. Perform an SGD update with momentum on the server side.  If
           ``self._nesterov`` is ``True``, use the Nesterov variant.
        4. Save the updated parameters via ``set_model_params`` and return them.
        
        self.log("aggregate: start")

        # Step-1: FedAvg aggregation (reuse FedMLAggOperator)
        aggregated_params: OrderedDict = FedMLAggOperator.agg(self.args, raw_client_model_list)

        # Step-2: Load current global params (on CPU)
        global_params: OrderedDict = self.get_model_params()

        # Step-3: Momentum update
        updated_params: OrderedDict = OrderedDict()
        for name, global_tensor in global_params.items():
            # Non-floating tensors (e.g. buffers) are copied directly
            if not torch.is_floating_point(global_tensor):
                updated_params[name] = aggregated_params[name]
                continue

            device = global_tensor.device           # cuda:0 (or cpu)
            agg_tensor = aggregated_params[name].to(device)
            grad = global_tensor - agg_tensor

            # Initialise velocity buffer if first time
            if name not in self._velocity:
                self._velocity[name] = torch.zeros_like(grad)

            # Momentum accumulation
            self._velocity[name] = self._momentum * self._velocity[name] + grad

            # Nesterov look-ahead
            if self._nesterov:
                update = self._momentum * self._velocity[name] + grad
            else:
                update = self._velocity[name]

            # Parameter update (SGD step)
            updated_params[name] = global_tensor - self._server_lr * update

        # Step-4: Push new params to the model & return
        self.set_model_params(updated_params)
        self.log("aggregate: finished")
        return updated_params
    """



class TrainingMetricsLogger:
    """Comprehensive logging for GRPO training with WandB support"""

    def __init__(self, log_dir: str, run_name: Optional[str] = None, 
                 enable_wandb: bool = False,
                 wandb_project: Optional[str] = None, wandb_entity: Optional[str] = None,
                 wandb_config: Optional[dict] = None):
        self.log_dir = log_dir
        self.run_name = run_name or f"grpo_training_{int(time.time())}"
        self.enable_wandb = enable_wandb

        # WandB setup
        self.wandb_run = None
        if self.enable_wandb:
            self.wandb_run = wandb.init(
                project=wandb_project or "grpo-training",
                entity=wandb_entity,
                name=self.run_name,
                config=wandb_config or {},
                reinit=True
            )
            print(f"WandB logging initialized. Project: {wandb_project or 'grpo-training'}")

        # Metrics tracking
        self.step_count = 0
        self.training_start_time = time.time()
        self.last_log_time = time.time()

        # Accumulated metrics for averaging
        self.accumulated_metrics = {
            'losses': [],
            'rewards': [],
            'kl_divergences': [],
            'policy_losses': [],
            'value_losses': [],
            'advantages': [],
            'rollout_lengths': []
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
            self.wandb_run.log(wandb_metrics, step=global_step)

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

    def log_moving_averages(self, global_step: int, window_size: int = 100):
        """Log moving averages of key metrics"""
        wandb_metrics = {}

        def get_moving_average(values, window):
            if len(values) == 0:
                return 0
            window = min(window, len(values))
            return sum(values[-window:]) / window

        # Moving averages
        if self.accumulated_metrics['losses']:
            avg_loss = get_moving_average(self.accumulated_metrics['losses'], window_size)
            wandb_metrics[f'moving_avg/loss_{window_size}'] = avg_loss

        if self.accumulated_metrics['rewards']:
            avg_reward = get_moving_average(self.accumulated_metrics['rewards'], window_size)
            wandb_metrics[f'moving_avg/reward_{window_size}'] = avg_reward

        if self.accumulated_metrics['kl_divergences']:
            avg_kl = get_moving_average(self.accumulated_metrics['kl_divergences'], window_size)
            wandb_metrics[f'moving_avg/kl_divergence_{window_size}'] = avg_kl

        if self.accumulated_metrics['rollout_lengths']:
            avg_length = get_moving_average(self.accumulated_metrics['rollout_lengths'], window_size)
            wandb_metrics[f'moving_avg/rollout_length_{window_size}'] = avg_length

        # Log to wandb
        if self.enable_wandb and wandb_metrics:
            self.wandb_run.log(wandb_metrics, step=global_step)

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

            self.logger.log_moving_averages(state.global_step, window_size=100)