"""
Adapted from https://github.com/lm-sys/FastChat/blob/af4dfe3f0ed481700265914af61b86e0856ac2d9/fastchat/protocol/openai_api_protocol.py
"""

import time
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal

import shortuuid
from pydantic import BaseModel, Field, validator

from ..utils import random_uuid


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fedml"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    @validator("n")
    def validate_n(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError(f"n must be at least 1 but got {v}")
        return v

    @validator("stop")
    def validate_stop(cls, v: Optional[Union[str, List[str]]]) -> List[str]:
        if v is None:
            v = []
        elif isinstance(v, str):
            v = [v]

        return [s for s in v if len(s) > 0]

    @validator("messages")
    def validate_messages(cls, v: Union[str, List[Dict[str, str]]]) -> List[str]:
        if isinstance(v, str):
            v = [{"role": "user", "content": v}]
        elif isinstance(v, list):
            for d in v:
                # force `role` to be lower case
                d["role"] = d["role"].lower()

        return v


class HFChatCompletionRequest(ChatCompletionRequest):
    # see https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/text_generation#transformers.GenerationConfig
    do_sample: Optional[bool] = False
    num_beams: Optional[int] = 1
    top_k: Optional[int] = 50
    min_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = 1.0

    def hf_dict(self) -> Dict[str, Any]:
        output_dict = self.dict()

        # translate keys
        output_dict["max_new_tokens"] = output_dict.pop("max_tokens", self.max_tokens)
        output_dict["num_return_sequences"] = output_dict.pop("n", self.n)
        # TODO: verify key type
        if self.logit_bias is not None:
            output_dict["sequence_bias"] = {
                (int(k),): v for k, v in output_dict.pop("logit_bias", self.logit_bias).items()
            }

        return output_dict


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None, description="data about request and response")


class CompletionRequest(BaseModel):
    model: str
    # a string, array of strings, array of tokens, or array of token arrays
    prompt: Union[List[int], List[List[int]], str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None  # OpenAI use 16. We use `None` to auto infer the value
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    @validator("n")
    def validate_n(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError(f"n must be at least 1 but got {v}")
        return v

    @validator("stop")
    def validate_stop(cls, v: Optional[Union[str, List[str]]]) -> List[str]:
        if v is None:
            v = []
        elif isinstance(v, str):
            v = [v]

        return [s for s in v if len(s) > 0]


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{shortuuid.random()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo]
