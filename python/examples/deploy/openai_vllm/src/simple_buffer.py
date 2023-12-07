from typing import Any, Dict, List, Optional, Sequence

from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np

from .typing import TokenizerType


class SimpleConversationBufferMemory(ConversationBufferMemory):
    human_prefix: str = "User"
    ai_prefix: str = "Assistant"
    memory_key: str = "history"

    human_prefix_key: str = "user"  # used to get human message from message dict
    ai_prefix_key: str = "assistant"  # used to get AI message from message dict
    human_message_prefix: str = ""
    human_message_suffix: str = ""
    ai_message_prefix: str = ""
    ai_message_suffix: str = ""

    max_tokens: int = 0  # set to non-positive number to disable it
    tokenizer: Optional[TokenizerType] = None

    @property
    def buffer_as_str(self) -> str:
        return "\n".join(self.buffer_as_strs)

    @property
    def buffer_as_strs(self) -> List[str]:
        string_messages = []
        for m in self.chat_memory.messages:
            if isinstance(m, HumanMessage):
                role = self.human_prefix
            elif isinstance(m, AIMessage):
                role = self.ai_prefix
            elif isinstance(m, SystemMessage):
                role = "System"
            elif isinstance(m, FunctionMessage):
                role = "Function"
            elif isinstance(m, ChatMessage):
                role = m.role
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            message = f"{role}: {m.content}" if len(role) > 0 else m.content
            if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
                message += f"{m.additional_kwargs['function_call']}"
            string_messages.append(message)

        return string_messages

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer. Pruned."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.add_user_message(input_str)
        self.add_ai_message(output_str)

        # Prune buffer if it exceeds max token limit
        self.prune_context()

    def add_user_message(self, message: str) -> None:
        # add prefix and suffix
        prefix = suffix = ""

        if not message.startswith(self.human_message_prefix):
            prefix = self.human_message_prefix

        if not message.endswith(self.human_message_suffix):
            suffix = self.human_message_suffix

        self.chat_memory.add_user_message(f"{prefix}{message}{suffix}")

    def add_ai_message(self, message: str) -> None:
        # add prefix and suffix
        prefix = suffix = ""

        if not message.startswith(self.ai_message_prefix):
            prefix = self.ai_message_prefix

        if not message.endswith(self.ai_message_suffix):
            suffix = self.ai_message_suffix

        self.chat_memory.add_ai_message(f"{prefix}{message}{suffix}")

    def add_message_dicts(self, message_dicts: Sequence[Dict[str, str]]) -> None:
        for idx, message in enumerate(message_dicts):
            # Each message should have `role` and `content` keys `role` is either "user" or "assistant"
            role = message["role"]
            content = message["content"]

            if role == self.human_prefix_key:
                self.add_user_message(content)
            elif role == self.ai_prefix_key:
                self.add_ai_message(content)
            else:
                raise ValueError(f"Unsupported role: {role}.")

        self.prune_context()

    def prune_context(self) -> None:
        if (
                self.max_tokens <= 0 or
                self.tokenizer is None or
                len(self.chat_memory.messages) == 0
        ):
            return

        buffer_str_lengths = [
            len(o) for o in self.tokenizer(self.buffer_as_strs, add_special_tokens=False)["input_ids"]
        ]

        # compute cumulative sum from the last element towards the first element
        buffer_cum_sizes = np.flip(np.cumsum(np.flip(buffer_str_lengths)))
        # find the indices where the cumsum is less than max_tokens
        buffer_start_indices = np.argwhere(buffer_cum_sizes <= self.max_tokens).flatten()
        if len(buffer_start_indices) == 0:
            self.clear()
        elif buffer_start_indices[0] > 0:
            # remove all elements before `buffer_start_indices`
            del self.chat_memory.messages[:buffer_start_indices[0]]
