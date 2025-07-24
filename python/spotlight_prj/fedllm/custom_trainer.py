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

from data_formatting import DataFormatting
from evaluation import Evaluation


class RewardFunction:

    def __init__(self, exact_match_reward, numeric_equivalence_reward, incorrect_answer_reward):

        self.exact_match_reward = exact_match_reward
        self.numeric_equivalence_reward = numeric_equivalence_reward
        self.incorrect_answer_reward = incorrect_answer_reward
        self.dat_fmt = DataFormatting()
        self.eval = Evaluation()


        pass

    def correctness_reward(self, completions, answer, **kwargs):

        """
        Assings a reward based on the correctness of the model's answer.

        Args:
            prompts (list): A list of input prompts.
            completons (list): List of model completions, each containing content.
            answer (list): List of expected answers. 
            **kwargs**: Additional keyword arguments.

        Returns:
            list: List of numerical rewards for each completion. 

        Explanation:
            1. Extracts content from each completion. 
            2. Extracts the answer portion from each response using extrac_answer_from_response
            3. Assigns rewards based on matching criteria:
                - 2.0 points for an exact match
                - 1.5 points for numeric equivalence (when values match but format differs)
                - 0.0 points for incorrect answers
            4. Tracks completion lengths for analysis.  
        """

        rewards = []

        for c, a in zip(completions, answer):

            if c==a: # exact match case
                rewards.append(self.exact_match_reward)

            else:
                #Try numeric equivalence
                c_num  = self.eval.extract_single_number(str(c))
                a_num = self.eval.extract_single_number(str(a))

                if c_num is not None and a_num is not None and c_num==a_num:

                    rewards.append(self.numeric_equivalence_reward)

                else:
                    rewards.append(self.incorrect_answer_reward)

        return rewards

    

    def combined_reward(self, completions, answer, **_):

        """
        Combines correctness and format rewards.

        Args:
            prompts (list[str]): List of prompt texts
            completions (list[list[dict]]): List of completion dictionaries.
            answer (list[str]): List of expected answers
        
        Returns:
            list[float]:Combined rewards for each prompt-completion pair
        
        Explanation:
            1. Calculates separate reward for correctness and format compliance.
            2. Combines the rewards with the following weights:
                - correctness score range: 0.0 to 2.0
                - Format score range 0.0 to 0.8
                - Total possible range: 0.0 to 2.8
            3. Returns the combined reward for each example. 
        """

        # Get individual rewards

        correctness_scores = self.correctness_reward(completions=completions,answer=answer)

        combined_reward = []

        for c_score in correctness_scores:

            combined_reward.append(c_score)


        return combined_reward

class TimedGRPOTrainer(GRPOTrainer):
    def _make_experience(self, *args, **kwargs):
        
        t0 = time.perf_counter()
        result = super()._make_experience(*args, **kwargs)
        self.log(f"roll-out batch {self.state.global_step} : "
                     f"{time.perf_counter() - t0:.3f}s")
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

        # ------------------------------------------------------------------
        # Configuration: enable or disable per-round checkpoints
        # ------------------------------------------------------------------

        # Default: omit per-round checkpoints unless user explicitly enables
        # them via the FedML YAML (enable_round_checkpoints: true)
        self._enable_round_ckpt = getattr(self.args, "enable_round_checkpoints", False)

        self.exact_match_reward = 2.0
        self.numeric_equivalence_reward=1.5
        self.incorrect_answer_reward=0.0
        self.rwdfn = RewardFunction(self.exact_match_reward, self.numeric_equivalence_reward, self.incorrect_answer_reward)
    
    def reward_fn(self, completions, answer, **_):
        """Reward function for GSM8K that checks if the predicted answer matches the true answer."""
        out = []
        for c, ans in zip(completions, answer):
            if c==ans:
                out.append(self.exact_match_reward)
            else:
                # Extract from dataset answer (GSM8K format)
                tru = self.DATASET_ANS.search(ans)
                # Extract from model completion (boxed format, fallback to GSM8K format)
                pred = self.MODEL_ANS.search(c)
                if not pred:
                    pred = self.DATASET_ANS.search(c)
                
                if pred and tru:
                    pred_num = pred.group(1)
                    tru_num = tru.group(1)
                    out.append(self.numeric_equivalence_reward if pred_num == tru_num else self.incorrect_answer_reward)
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
            fresh_model.load_state_dict(current_state, strict=False)
        
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
            max_completion_length=1024,
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
            report_to="wandb",
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
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": fresh_tokenizer.eos_token_id,
            "eos_token_id": fresh_tokenizer.eos_token_id,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.1,  # Prevent repetition
            "length_penalty": 1.0,      # Neutral length penalty
        }
        
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
                save_checkpoint(
                    self.model,
                    checkpoint_dir=ckpt_dir,
                    is_saving_process=self.training_args.should_save,
                    synchronize=True,
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

    def aggregate(self, raw_client_model_list):
        """Aggregate client models with Nesterov momentum.

        Steps
        -----
        1. Compute the FedAvg-style weighted average of client models (same as the
           default FedML behaviour).
        2. Treat the *difference* between the current global model and the
           aggregated model as the (negative) gradient.
        3. Perform an SGD update with momentum on the server side.  If
           ``self._nesterov`` is ``True``, use the Nesterov variant.
        4. Save the updated parameters via ``set_model_params`` and return them.
        """
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