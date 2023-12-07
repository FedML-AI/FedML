from typing import Any, AsyncGenerator, Dict, Optional, Union

import asyncio
from dataclasses import asdict
from http import HTTPStatus
import inspect
from pprint import pformat
import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fedml.serving import FedMLInferenceRunner, FedMLPredictor
import torch.cuda
import uvicorn
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

from src.model_builder import ModelArguments
from src.openai_api_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
    VLLMChatCompletionRequest,
)


class VLLMOpenAIChatCompletionInferenceRunner(FedMLInferenceRunner):
    def run(self) -> None:
        api = FastAPI()

        @api.post("/predict")
        @api.post("/chat/completions")
        async def predict(request: Request):
            input_json = await request.json()
            header = request.headers.get("Accept", "application/json")
            if header == "application/json" or header == "*/*":
                if input_json.get("stream", False):
                    resp = self.client_predictor.async_predict(input_json)
                    if asyncio.iscoroutine(resp):
                        resp = await resp

                    if isinstance(resp, Response):
                        return resp
                    else:
                        return StreamingResponse(resp)
                else:
                    return await self.client_predictor.async_predict(input_json)
            else:
                response_obj = self.client_predictor.predict(input_json, request.headers.get("Accept"))
                return FileResponse(response_obj)

        @api.get("/ready")
        async def ready():
            return {"status": "Success"}

        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)


# Adapted from Adapted from https://github.com/vllm-project/vllm/blob/7e90a2d11785b4cba5172f13178adb6d194a867f/vllm/entrypoints/api_server.py
class VLLMOpenAIChatbot(FedMLPredictor):
    SAMPLING_PARAMS_ALLOWED_KEYS = set(inspect.signature(SamplingParams.__init__).parameters.keys())

    def __init__(self, model_args: Optional[ModelArguments] = None):
        super().__init__()

        if model_args is None:
            model_args = ModelArguments.from_environ(
                default_kwargs=dict(
                    model_name_or_path="~/fedml_serving/model_and_config",
                    top_k=0,
                    top_p=0.92,
                    verbose=True
                )
            )
        self.model_args = model_args

        if self.model_args.verbose:
            print(f"model config: {pformat(asdict(self.model_args))}")

        self.default_sampling_params_kwargs = dict(
            temperature=self.model_args.temperature,
            top_k=self.model_args.top_k if self.model_args.top_k > 0 else -1,
            top_p=self.model_args.top_p,
            max_tokens=self.model_args.max_new_tokens
        )

        engine_args = AsyncEngineArgs(
            model=model_args.model_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = get_tokenizer(
            tokenizer_name=engine_args.tokenizer,
            tokenizer_revision=engine_args.tokenizer_revision,
            trust_remote_code=engine_args.trust_remote_code
        )

    async def async_predict(self, request_dict: Dict[str, Any]) -> Union[StreamingResponse, Dict[str, Any]]:
        created_time = int(time.monotonic())

        request_id = request_dict.get("request_id", random_uuid())
        max_model_length = (await self.engine.get_model_config()).max_model_len

        request = VLLMChatCompletionRequest(**request_dict)

        # see https://huggingface.co/docs/transformers/v4.34.1/chat_templating
        input_str = self.tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(input_str).input_ids

        num_input_tokens = len(input_ids)
        if request.max_tokens is None:
            request.max_tokens = max_model_length - num_input_tokens
        if num_input_tokens + request.max_tokens > max_model_length:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"This model's maximum context length is {max_model_length} tokens but"
                       f" {request.max_tokens + num_input_tokens} tokens are required"
                       f"({num_input_tokens} for input, "
                       f"{request.max_tokens} for completion). "
                       f"Please reduce the length of the messages or completion."
            )

        sampling_param_kwargs = self.default_sampling_params_kwargs.copy()
        sampling_param_kwargs.update(request.dict())
        sampling_params = SamplingParams(**{
            k: v for k, v in sampling_param_kwargs.items() if k in self.SAMPLING_PARAMS_ALLOWED_KEYS
        })

        print(f"Processing request: {request_id}", flush=True)
        chat_generator = self.engine.generate(input_str, sampling_params, request_id)

        # Streaming response
        if request.stream:
            return StreamingResponse(
                self.chat_completion_stream_generator(
                    chat_generator,  # noqa
                    request_id=request_id,
                    n=sampling_params.n,
                    created_time=created_time
                ), media_type="text/event-stream"
            )
        else:
            return (await self.completion_full_generator(
                chat_generator,  # noqa
                request_id=request_id,
                created_time=created_time
            )).dict()

    async def chat_completion_stream_generator(
            self,
            chat_generator: AsyncGenerator[RequestOutput, None],
            request_id: str,
            n: int,
            created_time: int,
            model_name: str = "local_model",
            chunk_object_type: str = "chat.completion.chunk"
    ) -> AsyncGenerator[bytes, None]:
        for i in range(n):
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    ),
                ],
                model=model_name
            )
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

            # Send response for each token for each request.n (index)
            previous_texts = [""] * n
            previous_num_tokens = [0] * n
            finish_reason_sent = [False] * n
            async for res in chat_generator:
                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        delta_text = output.text[len(previous_texts[i]):]
                        previous_texts[i] = output.text
                        completion_tokens = len(output.token_ids)
                        previous_num_tokens[i] = completion_tokens
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=delta_text),
                                    finish_reason=None
                                ),
                            ],
                            model=model_name)
                        data = chunk.json(exclude_unset=True, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    else:
                        # Send the finish response for each request.n only once
                        prompt_tokens = len(res.prompt_token_ids)
                        completion_tokens = previous_num_tokens[i]
                        final_usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(),
                                    finish_reason=output.finish_reason
                                ),
                            ],
                            model=model_name
                        )
                        if final_usage is not None:
                            chunk.usage = final_usage
                        data = chunk.json(exclude_unset=True,
                                          exclude_none=True,
                                          ensure_ascii=False)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True
            # Send the final done message after all response.n are finished
            yield "data: [DONE]\n\n"

    async def completion_full_generator(
            self,
            chat_generator: AsyncGenerator[RequestOutput, None],
            request_id: str,
            created_time: int,
            model_name: str = "local_model",
            raw_request: Optional[Request] = None
    ) -> ChatCompletionResponse:
        final_res: Optional[RequestOutput] = None
        async for res in chat_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="Client disconnected"
                )

            final_res = res
        assert final_res is not None

        choices = []
        role = "assistant"
        for output in final_res.outputs:
            choices.append(ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason=output.finish_reason,
            ))

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response


if __name__ == "__main__":
    chatbot = VLLMOpenAIChatbot()
    fedml_inference_runner = VLLMOpenAIChatCompletionInferenceRunner(chatbot)
    fedml_inference_runner.run()
