from typing import Any, Dict, Optional, Tuple

from abc import abstractmethod
from dataclasses import asdict
from pprint import pformat

from fedml.serving import FedMLPredictor
from langchain import PromptTemplate

from .model_builder import ModelArguments
from .simple_buffer import SimpleConversationBufferMemory
from .typing import TokenizerType


class BaseChatbot(FedMLPredictor):
    support_async_predict: bool = False

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

        prompt_info = self.model_args.prompt_info
        self.prompt = PromptTemplate(
            input_variables=prompt_info.input_variables,
            template=prompt_info.prompt_template
        )
        self.memory = SimpleConversationBufferMemory(
            human_prefix=prompt_info.human_key,
            ai_prefix=prompt_info.ai_key,
            memory_key=prompt_info.memory_key,
            human_message_prefix=prompt_info.human_message_prefix,
            human_message_suffix=prompt_info.human_message_suffix,
            ai_message_prefix=prompt_info.ai_message_prefix,
            ai_message_suffix=prompt_info.ai_message_suffix,
            max_tokens=self.model_args.max_history_tokens
        )

    @property
    def tokenizer(self) -> TokenizerType:
        return self.memory.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerType) -> None:
        self.memory.tokenizer = tokenizer

    def predict(self, request: Dict[str, Any]) -> str:
        args, kwargs = self.parse_input(request)
        return str(self(*args, **kwargs))

    def parse_input(self, request: Dict[str, Any]) -> Tuple[tuple, Dict[str, Any]]:
        # The request format follows https://platform.openai.com/docs/guides/gpt/chat-completions-api
        if "messages" in request:
            messages = request["messages"]
        elif "inputs" in request:
            # TODO: remove this temporary fix
            messages = request["inputs"].get("messages", [])
        else:
            messages = []

        if len(messages) == 0:
            raise ValueError("Received empty input.")

        # remove chat history
        self.clear_history()

        last_message, history = messages[-1], messages[:-1]

        question = last_message["content"]
        question_role = last_message["role"]
        if question_role != "user":
            raise ValueError(f"The last message should be a \"user\" message but has role \"{question_role}\".")

        unsupported_roles = [m for m in history if m.get("role", "") not in ("user", "assistant")]
        if len(unsupported_roles) > 0:
            raise ValueError(f"Unsupported roles {unsupported_roles} found in request.")

        self.memory.add_message_dicts(history)

        # add message prefix and suffix
        prompt_info = self.model_args.prompt_info
        question = f"{prompt_info.human_message_prefix}{question}{prompt_info.human_message_suffix}"
        return (question,), {}

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass

    def clear_history(self) -> None:
        self.memory.clear()
