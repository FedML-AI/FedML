"""
References
    - https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming
    - https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextStreamer
    - https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextIteratorStreamer
    - https://github.com/huggingface/transformers/issues/23933
    - https://github.com/flozi00/atra/blob/375bd740c37fb42d35048ae33ae414841f22938a/atra/text_utils/chat.py#LL98C7-L98C7
"""

import asyncio
from copy import deepcopy
import json
from functools import partial
from http import HTTPStatus
from pprint import pformat
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Union

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fedml.serving import FedMLInferenceRunner
from transformers import BatchEncoding, StoppingCriteriaList

from src.base_chatbot import BaseChatbot
from src.modeling_utils import get_max_seq_length, StopStringCriteria
from src.protocol import LLMInput, LLMOutput
from src.streamers import AsyncTextIteratorStreamer


class HFChatbot(BaseChatbot):
    def __init__(self):
        super().__init__()

        self.llm = self.model_args.get_hf_pipeline(
            task="text-generation",
            trust_remote_code=True,
            return_full_text=False
        )

        self.tokenizer = self.llm.tokenizer
        if self.model_args.chat_template is not None:
            self.tokenizer.chat_template = self.model_args.chat_template
        if self.tokenizer.chat_template is None and self.model_args.default_chat_template is not None:
            self.tokenizer.chat_template = self.model_args.default_chat_template

        self.default_generation_config = deepcopy(self.llm.model.generation_config)
        self.default_generation_config.update(**self.model_args.generation_kwargs)
        self.default_generation_config.validate()

    def preprocess(self, prompt: str, request_dict: Dict[str, Any]) -> LLMInput:
        inputs: BatchEncoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            # models such as llama and mistral already added special tokens to the prompt (e.g. prepends BOS token).
            #   skip adding special tokens if needed.
            add_special_tokens=not prompt.startswith(self.tokenizer.bos_token)
        )
        assert len(inputs.input_ids) == 1  # TODO: improve error message

        # (batch_size, sequence_length)
        num_prompt_tokens = inputs.input_ids.shape[1]
        max_model_length = get_max_seq_length(self.llm.model)

        if request_dict.get("max_new_tokens", None) is None:
            max_new_tokens = max_model_length - num_prompt_tokens
        else:
            max_new_tokens = request_dict["max_new_tokens"]
        if num_prompt_tokens + max_new_tokens > max_model_length:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"This model's maximum context length is {max_model_length} tokens but"
                       f" {max_new_tokens + num_prompt_tokens} tokens are required"
                       f"({num_prompt_tokens} for input, "
                       f"{max_new_tokens} for completion). "
                       f"Please reduce the length of the messages or completion."
            )

        generation_config = deepcopy(self.default_generation_config)
        generation_config.update(**request_dict)
        generation_config.update(max_new_tokens=max_new_tokens)
        generation_config.validate()

        stopping_criteria = None
        stop: Optional[Union[List[str], str]] = request_dict.get("stop", None)
        if bool(stop):
            if isinstance(stop, str):
                stop = [stop]

            if any(len(s) > 0 for s in stop):
                stop_string_criteria = StopStringCriteria(self.tokenizer, stop, num_prompt_tokens)
                stop = list(stop_string_criteria.stop)

                stopping_criteria = StoppingCriteriaList([
                    stop_string_criteria,
                ])

        if self.model_args.verbose:
            print(
                f"Stop strings: {stop}\n"
                f"Generation config: {pformat(generation_config)}\n"
                f"Prompt after formatting:\n"
                f"==============================\n"
                f"{prompt}\n"
                f"=============================="
            )

        return LLMInput(
            prompt=prompt,
            inputs=inputs,
            num_prompt_tokens=num_prompt_tokens,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            stop=stop
        )

    def predict(self, request_dict: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        model_input = self.preprocess(
            self.apply_chat_template(
                request_dict["messages"],
                tokenize=False,
                add_generation_prompt=True
            ),
            request_dict
        )

        model_output = self.generate(model_input)

        if isinstance(model_output, Sequence) and len(model_output) == 1:
            return model_output[0].to_dict()
        else:
            return [o.to_dict() for o in model_output]

    def generate(self, model_input: LLMInput) -> List[LLMOutput]:
        model_output: List[Dict[str, Any]] = self.llm(
            model_input.prompt,
            generation_config=model_input.generation_config,
            stopping_criteria=model_input.stopping_criteria
        )

        outputs = []
        for output_dict in model_output:
            # remove stop strings
            if bool(model_input.stop):
                generated_text = output_dict["generated_text"]

                for s in model_input.stop:
                    if generated_text.endswith(s):
                        output_dict["generated_text"] = generated_text[:-len(s)]
                        break

            outputs.append(LLMOutput.from_dict(output_dict, self.tokenizer))

        return outputs

    async def async_predict(self, request_dict: Dict[str, Any]) -> StreamingResponse:
        model_input = self.preprocess(
            self.apply_chat_template(
                request_dict["messages"],
                tokenize=False,
                add_generation_prompt=True
            ),
            request_dict
        )

        async def text_stream_generator() -> AsyncGenerator[bytes, None]:
            async for output in self.generate_stream(model_input):
                output_dict = {"generated_text": output.text}
                yield (json.dumps(output_dict) + "\n").encode("utf-8")

        return StreamingResponse(text_stream_generator(), media_type="application/json")

    async def generate_stream(self, model_input: LLMInput) -> AsyncGenerator[LLMOutput, None]:
        # see https://stackoverflow.com/a/43263397
        # see https://superfastpython.com/what-is-asyncio-sleep-zero/
        loop = asyncio.get_running_loop()

        streamer = AsyncTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            stop=model_input.stop,
            verbose=self.model_args.verbose,
            skip_special_tokens=True
        )
        future = loop.run_in_executor(
            None,
            partial(
                self.llm.model.generate,
                # move to correct device
                **self.llm.ensure_tensor_on_device(**model_input.inputs),
                streamer=streamer,
                generation_config=model_input.generation_config,
                stopping_criteria=model_input.stopping_criteria
            )
        )

        async for streamer_output in streamer:
            if self.model_args.verbose:
                print(f"[Stream]: {streamer_output}", flush=True)
            yield streamer_output

        await future


if __name__ == "__main__":
    chatbot = HFChatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
