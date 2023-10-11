"""
Adapted from https://github.com/databrickslabs/dolly/blob/master/training/generate.py
"""
from typing import List, Optional, Tuple
import logging
import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

from .constants import END_KEY, PROMPT_FOR_GENERATION_FORMAT, RESPONSE_KEY

logger = logging.getLogger(__name__)


def load_model_tokenizer_for_generate(
        pretrained_model_name_or_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the model and tokenizer for generating responses.

    Args:
        pretrained_model_name_or_path (str): Name or path for the pretrained model.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.

    Example:
        model, tokenizer = load_model_tokenizer_for_generate("gpt2")
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    return model, tokenizer

def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """
    Get the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token. This function retrieves the token ID for a given key.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer.
        key (str): The key to convert to a single token.

    Raises:
        ValueError: If more than one ID was generated for the key.

    Returns:
        int: The token ID for the given key.

    Example:
        special_token_id = get_special_token_id(tokenizer, "### Instruction:")
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]



class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
            self,
            *args,
            do_sample: bool = True,
            max_new_tokens: int = 256,
            top_p: float = 0.92,
            top_k: int = 0,
            **kwargs
    ):
        """Initialize the pipeline

        Args:
            do_sample (bool, optional): Whether to use sampling. Defaults to True.
            max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
            top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
                probabilities that add up to top_p or higher are kept for generation. Defaults to 0.92.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
                Defaults to 0.
        """
        super().__init__(
            *args,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )

    def _sanitize_parameters(
            self,
            return_full_text: bool = None,
            **generate_kwargs
    ):
        """
        Sanitize and configure parameters for text generation.

        Args:
            return_full_text (bool, optional): Whether to return the full text. Defaults to None.

        Returns:
            Tuple[Dict, Dict, Dict]: A tuple containing preprocess_params, forward_params, and postprocess_params.

        Raises:
            ValueError: If the response key token is not found.
        """
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id
        }

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        """
        Preprocess the input text for text generation.

        Args:
            instruction_text (str): The instruction text.

        Returns:
            Dict: Preprocessed inputs for text generation.

        Example:
            inputs = preprocess("Write a summary of a book.")
        """
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        """
        Forward pass for text generation.

        Args:
            model_inputs (Dict): Inputs for text generation.

        Returns:
            Dict: Model outputs for text generation.
        """
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)

        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generated_sequence = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))

        instruction_text = model_inputs.pop("instruction_text")
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "instruction_text": instruction_text}

    def postprocess(
            self,
            model_outputs,
            response_key_token_id: Optional[int] = None,
            end_key_token_id: Optional[int] = None,
            return_full_text: bool = False
    ):
        """
        Postprocess the model outputs for text generation.

        Args:
            model_outputs (Dict): Model outputs for text generation.
            response_key_token_id (int, optional): Token ID for the response key. Defaults to None.
            end_key_token_id (int, optional): Token ID for the end key. Defaults to None.
            return_full_text (bool, optional): Whether to return the full text. Defaults to False.

        Returns:
            List[Dict]: List of generated text records.

        Example:
            generated_text = postprocess(model_outputs)
        """
        generated_sequence: torch.Tensor = model_outputs["generated_sequence"][0]
        instruction_text = model_outputs["instruction_text"]

        generated_sequence: List[List[int]] = generated_sequence.tolist()
        records = []
        for sequence in generated_sequence:

            # The response will be set to this variable if we can identify it.
            decoded = None

            # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
            if response_key_token_id and end_key_token_id:
                # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
                # prompt, we should definitely find it.  We will return the tokens found after this token.
                try:
                    response_pos = sequence.index(response_key_token_id)
                except ValueError:
                    logger.warning(f"Could not find response key {response_key_token_id} in: {sequence}")
                    response_pos = None

                if response_pos:
                    # Next find where "### End" is located.  The model has been trained to end its responses with this
                    # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                    # this token, as the response could be truncated.  If we don't find it then just return everything
                    # to the end. Note that even though we set eos_token_id, we still see this token at the end.
                    try:
                        end_pos = sequence.index(end_key_token_id)
                    except ValueError:
                        end_pos = None

                    decoded = self.tokenizer.decode(sequence[response_pos + 1: end_pos]).strip()

            if not decoded:
                # Otherwise we'll decode everything and use a regex to find the response and end.

                fully_decoded = self.tokenizer.decode(sequence)

                # The response appears after "### Response:".  The model has been trained to append "### End" at the
                # end.
                m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)

                if m:
                    decoded = m.group(1).strip()
                else:
                    # The model might not generate the "### End" sequence before reaching the max tokens. In this case,
                    # return everything after "### Response:".
                    m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                    if m:
                        decoded = m.group(1).strip()
                    else:
                        logger.warning(f"Failed to find response in:\n{fully_decoded}")

            # If the full text is requested, then append the decoded text to the original instruction.
            # This technically isn't the full text, as we format the instruction in the prompt the model has been
            # trained on, but to the client it will appear to be the full text.
            if return_full_text:
                decoded = f"{instruction_text}\n{decoded}"

            rec = {"generated_text": decoded}

            records.append(rec)

        return records
    




def generate_response(
        instruction: str,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        **kwargs,
) -> str:
    """
    Given an instruction, uses the model and tokenizer to generate a response. This formats the instruction in
    the instruction format that the model was fine-tuned on.

    Args:
        instruction (str): The instruction for text generation.
        model (PreTrainedModel): The pretrained model to use for text generation.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the pretrained model.
        **kwargs: Additional keyword arguments for text generation.

    Returns:
        str: The generated response based on the provided instruction.

    Example:
        response = generate_response("Write a summary of a book.", model=my_model, tokenizer=my_tokenizer)
    """

    generation_pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer, **kwargs)
    return generation_pipeline(instruction)[0]["generated_text"]