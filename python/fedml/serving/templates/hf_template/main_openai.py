"""
References
    - https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming
    - https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextStreamer
    - https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextIteratorStreamer
    - https://github.com/huggingface/transformers/issues/23933
    - https://github.com/flozi00/atra/blob/375bd740c37fb42d35048ae33ae414841f22938a/atra/text_utils/chat.py#LL98C7-L98C7
"""

import asyncio
import warnings
from http import HTTPStatus
from pprint import pformat
import time
from typing import Any, AsyncGenerator, Dict, Sequence, Union

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fedml.serving import FedMLInferenceRunner
import uvicorn

from main import HFChatbot
from src.protocol import LLMInput
from src.protocol.openai import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    HFChatCompletionRequest,
    UsageInfo,
)
from src.utils import random_uuid


class HFOpenAIChatCompletionInferenceRunner(FedMLInferenceRunner):
    def run(self) -> None:
        api = FastAPI()

        @api.post("/predict")
        @api.post("/completions")
        @api.post("/chat/completions")
        async def predict(request: Request):
            input_json = await request.json()
            if input_json.get("stream", False):
                resp = self.client_predictor.async_predict(input_json)
                if asyncio.iscoroutine(resp):
                    resp = await resp

                if isinstance(resp, Response):
                    return resp
                else:
                    return StreamingResponse(resp, media_type="text/event-stream")
            else:
                return await self.client_predictor.async_predict(input_json)

        @api.get("/ready")
        async def ready():
            return {"status": "Success"}

        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)


class HFOpenAIChatbot(HFChatbot):

    async def async_predict(self, request_dict: Dict[str, Any]) -> Union[StreamingResponse, Dict[str, Any]]:
        if "messages" in request_dict:
            resp = self.create_chat_completion(request_dict)
        elif "prompt" in request_dict:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Completions API is not supported yet."
            )
        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Request must have either \"messages\" or \"prompt\"in the request body."
            )

        if asyncio.iscoroutine(resp):
            resp = await resp
        return resp

    async def create_chat_completion(self, request_dict: Dict[str, Any]) -> Union[StreamingResponse, Dict[str, Any]]:
        created_time = int(time.monotonic())

        request_id = request_dict.get("request_id", random_uuid())
        # inject default generation config
        request = HFChatCompletionRequest(**{**self.default_generation_config.to_diff_dict(), **request_dict})
        if self.model_args.verbose:
            print(f"\nRequest: {pformat(request)}", flush=True)

        # translate OpenAI keys to HF keys
        request_dict.update(request.hf_dict())

        model_input = self.preprocess(
            self.apply_chat_template(
                request_dict["messages"],
                tokenize=False,
                add_generation_prompt=True
            ),
            request_dict
        )

        if request.stream:
            return StreamingResponse(
                self.chat_completion_stream_generator(
                    model_input=model_input,
                    request_id=request_id,
                    request=request,
                    created_time=created_time
                ),
                media_type="text/event-stream"
            )
        else:
            return self.chat_completion_generate(
                model_input=model_input,
                request_id=request_id,
                request=request,
                created_time=created_time
            ).dict()

    async def chat_completion_stream_generator(
            self,
            model_input: LLMInput,
            request_id: str,
            request: HFChatCompletionRequest,
            created_time: int,
            model_name: str = "local_model",
            chunk_object_type: str = "chat.completion.chunk"
    ) -> AsyncGenerator[str, None]:
        if request.n > 1:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Streaming with n > 1 is currently not supported but received n = {request.n}."
            )

        # this should be replaced with a for loop if n > 1
        curr_index = 0

        chunk = ChatCompletionStreamResponse(
            id=request_id,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=curr_index,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                ),
            ],
            model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_num_tokens = 0

        async for output in self.generate_stream(model_input):
            previous_num_tokens += len(output.token_ids)

            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object=chunk_object_type,
                created=created_time,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=curr_index,
                        delta=DeltaMessage(content=output.text),
                        finish_reason=None
                    ),
                ],
                model=model_name
            )
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        completion_tokens = previous_num_tokens
        total_tokens = model_input.num_prompt_tokens + completion_tokens

        chunk = ChatCompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=curr_index,
                    delta=DeltaMessage(),
                    finish_reason="stop"  # TODO: support other `finish_reason`
                ),
            ],
            model=model_name,
            usage=UsageInfo(
                prompt_tokens=model_input.num_prompt_tokens,
                total_tokens=total_tokens,
                completion_tokens=completion_tokens
            )
        )

        yield f"data: {chunk.json(exclude_unset=True, exclude_none=True, ensure_ascii=False)}\n\n"

        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    def chat_completion_generate(
            self,
            model_input: LLMInput,
            request_id: str,
            request: HFChatCompletionRequest,
            created_time: int,
            model_name: str = "local_model"
    ) -> ChatCompletionResponse:
        if request.n > 1:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Non-streaming with n > 1 is currently unsupported but received n = {request.n}."
            )

        # this should be replaced with a for loop if n > 1
        curr_index = 0

        outputs = self.generate(model_input)
        if isinstance(outputs, Sequence) and len(outputs) > 1:
            warnings.warn(f"Multiple outputs is not supported. Only keeping the first output.")

        output = outputs[0]

        role = "assistant"
        choices = [
            ChatCompletionResponseChoice(
                index=curr_index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason="stop"  # TODO: support other `finish_reason`
            )
        ]

        completion_tokens = len(output.token_ids)
        total_tokens = model_input.num_prompt_tokens + completion_tokens

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=model_input.num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )


if __name__ == "__main__":
    chatbot = HFOpenAIChatbot()
    fedml_inference_runner = HFOpenAIChatCompletionInferenceRunner(chatbot)
    fedml_inference_runner.run()
