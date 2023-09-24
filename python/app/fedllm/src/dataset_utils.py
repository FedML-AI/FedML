from typing import Any, Callable, MutableMapping, TypeVar

from functools import partial

T = TypeVar("T")

# -----------------------------------------------------------------
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

# This is a training prompt that does not contain an input string. The instruction by itself has enough information
# to respond.For example, the instruction might ask for the year a historic figure was born.
DOLLY_PROMPT_TEMPLATE = f"""\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

{INSTRUCTION_KEY}
{{instruction}}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""

# This is a training prompt that contains an input string that serves as context for the instruction. For example,
# the input might be a passage from Wikipedia and the instruction is to extract some information from it.
DOLLY_PROMPT_WITH_CONTEXT_TEMPLATE = f"""\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

{INSTRUCTION_KEY}
{{instruction}}

{INPUT_KEY}
{{context}}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""


# Adapted from https://www.philschmid.de/deepspeed-lora-flash-attention
def format_dolly(sample: MutableMapping[str, Any]) -> str:
    context = sample.get("context", "")

    if context is not None and len(context) > 0:
        prompt_template = DOLLY_PROMPT_WITH_CONTEXT_TEMPLATE
    else:
        prompt_template = DOLLY_PROMPT_TEMPLATE
    return prompt_template.format(**sample)


# -----------------------------------------------------------------
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# see https://python.langchain.com/docs/modules/memory/adding_memory
# see https://stackoverflow.com/a/76936235
# see https://blog.futuresmart.ai/integrating-llama-2-with-hugging-face-and-langchain
# see https://github.com/facebookresearch/llama/issues/481
# see https://github.com/facebookresearch/llama/issues/484
# see https://github.com/facebookresearch/llama/issues/435
# this template is adapted from the HF example
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
LLAMA_PROMPT_TEMPLATE = f"""\
{B_INST} {B_SYS}
Below is an instruction that describes a task. Write a response that appropriately completes the request.
{E_SYS}

{INSTRUCTION_KEY}
{{instruction}}
{E_INST}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""

LLAMA_PROMPT_WITH_CONTEXT_TEMPLATE = f"""\
{B_INST} {B_SYS}
Below is an instruction that describes a task. Write a response that appropriately completes the request.
{E_SYS}

{INSTRUCTION_KEY}
{{instruction}}

{INPUT_KEY}
{{context}}
{E_INST}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""


def format_llama(sample: MutableMapping[str, Any]) -> str:
    context = sample.get("context", "")

    if context is not None and len(context) > 0:
        prompt_template = LLAMA_PROMPT_WITH_CONTEXT_TEMPLATE
    else:
        prompt_template = LLAMA_PROMPT_TEMPLATE
    return prompt_template.format(**sample)


# -----------------------------------------------------------------
def get_prompt_formatter(prompt_style: str) -> Callable[[MutableMapping[str, Any]], str]:
    if prompt_style == "llama":
        return partial(apply_prompt_template, template_func=format_llama)
    else:
        return partial(apply_prompt_template, template_func=format_dolly)


def apply_prompt_template(sample: T, template_func: Callable[[MutableMapping[str, Any]], str]) -> T:
    sample["text"] = template_func(sample)
    return sample
