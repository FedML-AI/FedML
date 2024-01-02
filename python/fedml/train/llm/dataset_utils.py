from typing import Any, Callable, Mapping, MutableMapping, NamedTuple, Optional, TypeVar

from functools import partial

T = TypeVar("T")
D = TypeVar(name="D", bound=MutableMapping[str, Any])

# -----------------------------------------------------------------
# This is a training prompt that does not contain an input string. The instruction by itself has enough information
# to respond.For example, the instruction might ask for the year a historic figure was born.
DEFAULT_PROMPT_TEMPLATE = f"""\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

### Response:
{{response}}
"""

# This is a training prompt that contains an input string that serves as context for the instruction. For example,
# the input might be a passage from Wikipedia and the instruction is to extract some information from it.
DEFAULT_PROMPT_WITH_CONTEXT_TEMPLATE = f"""\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response \
that appropriately completes the request.

### Instruction:
{{instruction}}

### Input:
{{context}}

### Response:
{{response}}
"""

# -----------------------------------------------------------------
DOLLY_PROMPT_TEMPLATE = f"""\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

### Response:
{{response}}

### End
"""

DOLLY_PROMPT_WITH_CONTEXT_TEMPLATE = f"""\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{instruction}}

Input:
{{context}}

### Response:
{{response}}

### End
"""

# -----------------------------------------------------------------
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# see https://python.langchain.com/docs/modules/memory/adding_memory
# see https://stackoverflow.com/a/76936235
# see https://blog.futuresmart.ai/integrating-llama-2-with-hugging-face-and-langchain
# see https://github.com/facebookresearch/llama/issues/481
# see https://github.com/facebookresearch/llama/issues/484
# see https://github.com/facebookresearch/llama/issues/435
# this template is adapted from the HF example
LLAMA_PROMPT_TEMPLATE = f"""\
[INST] <<SYS>>
Below is an instruction that describes a task. Write a response that appropriately completes the request.
<</SYS>>

{{instruction}}
[/INST]
{{response}}
"""

LLAMA_PROMPT_WITH_CONTEXT_TEMPLATE = f"""\
[INST] <<SYS>>
Below is an instruction that describes a task. Write a response that appropriately completes the request.
<</SYS>>

{{instruction}}

### Input:
{{context}}
[/INST]
{{response}}
"""


# -----------------------------------------------------------------

class PromptInfo(NamedTuple):
    prompt_template: str
    prompt_with_context_template: str

    # Adapted from https://www.philschmid.de/deepspeed-lora-flash-attention
    def apply_prompt_template(self, sample: D) -> D:
        context = sample.get("context", "")

        if context is not None and len(context) > 0:
            prompt_template = self.prompt_with_context_template
        else:
            prompt_template = self.prompt_template

        sample["text"] = prompt_template.format(**sample)
        return sample


PROMPT_TEMPLATE_MAPPING = {
    "default": PromptInfo(
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        prompt_with_context_template=DEFAULT_PROMPT_WITH_CONTEXT_TEMPLATE,
    ),
    "dolly": PromptInfo(
        prompt_template=DOLLY_PROMPT_TEMPLATE,
        prompt_with_context_template=DOLLY_PROMPT_WITH_CONTEXT_TEMPLATE,
    ),
    "llama": PromptInfo(
        prompt_template=LLAMA_PROMPT_TEMPLATE,
        prompt_with_context_template=LLAMA_PROMPT_WITH_CONTEXT_TEMPLATE,
    ),
}


def get_prompt_formatter(prompt_style: str) -> Callable[[MutableMapping[str, Any]], str]:
    prompt_info = PROMPT_TEMPLATE_MAPPING.get(prompt_style, PROMPT_TEMPLATE_MAPPING["default"])

    return prompt_info.apply_prompt_template


# -----------------------------------------------------------------

DEFAULT_COLUMN_NAME_MAPPING = {
    # for datasets such as yahma/alpaca-cleaned
    "input": "context",
    "output": "response",
    # for datasets such as tiiuae/falcon-refinedweb
    "content": "text",
}

DEFAULT_KEYWORD_REPLACEMENTS = {
    # for "lavita/ChatDoctor-HealthCareMagic-100k"
    "Chat Doctor": "AI Assistant",
    "ChatDoctor": "AI Assistant",
}


def get_keyword_replacer(replacements: Optional[Mapping[str, str]] = None) -> Callable[[MutableMapping[str, Any]], str]:
    if replacements is None:
        return partial(apply_keyword_replacer, replacements=DEFAULT_KEYWORD_REPLACEMENTS)

    else:
        return partial(apply_keyword_replacer, replacements=replacements)


def apply_keyword_replacer(sample: T, replacements: Mapping[str, str]) -> T:
    for keyword, replacement in replacements.items():
        sample["text"] = sample["text"].replace(keyword, replacement)
    return sample
