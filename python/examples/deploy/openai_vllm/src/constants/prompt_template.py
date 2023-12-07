from typing import NamedTuple, List

from string import Formatter

# -----------------------------------------------------------------
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

# -----------------------------------------------------------------
DEFAULT_PROMPT_TEMPLATE = f"""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as \
possible, while being safe. Your answers should not include any harmful, unethical\
, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses \
are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead \
of answering something not correct. If you don't know the answer to a question, please don't \
share false information.

Previous Conversation:
'''
{{history}}
'''

{INSTRUCTION_KEY}
User: {{input}}

{RESPONSE_KEY}
Assistant:
"""

# -----------------------------------------------------------------
DOLLY_PROMPT_TEMPLATE = f"""\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

You are a helpful, respectful and honest assistant. Always answer as helpfully as \
possible, while being safe. Your answers should not include any harmful, unethical\
, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses \
are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead \
of answering something not correct. If you don't know the answer to a question, please don't \
share false information.

Previous Conversation:
'''
{{history}}
'''

{INSTRUCTION_KEY}
User: {{input}}

{RESPONSE_KEY}
Assistant:
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
You are a helpful, respectful and honest assistant. Always answer as helpfully as \
possible, while being safe. Your answers should not include any harmful, unethical\
, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses \
are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead \
of answering something not correct. If you don't know the answer to a question, please don't \
share false information.
<</SYS>> [/INST]

{{history}}

{{input}}
"""

# -----------------------------------------------------------------
# See https://huggingface.co/OpenAssistant/llama2-13b-orca-8k-3319
# See https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
OASST_PROMPT_TEMPLATE = f"""\
<|system|>
Below is an instruction that describes a task. Write a response that appropriately completes the request.

You are a helpful, respectful and honest assistant. Always answer as helpfully as \
possible, while being safe. Your answers should not include any harmful, unethical\
, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses \
are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead \
of answering something not correct. If you don't know the answer to a question, please don't \
share false information.
</s>
{{history}}
{{input}}
<|assistant|>
"""

# -----------------------------------------------------------------
# See https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
# See https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
ZEPHYR_PROMPT_TEMPLATE = OASST_PROMPT_TEMPLATE


# -----------------------------------------------------------------
class PromptInfo(NamedTuple):
    prompt_template: str
    input_key: str = "input"
    memory_key: str = "history"
    # Role name for user
    human_key: str = "User"
    # Role name for AI assistant
    ai_key: str = "Assistant"
    # prefix to prepend the input human message
    human_message_prefix: str = ""
    human_message_suffix: str = ""
    ai_message_prefix: str = ""
    ai_message_suffix: str = ""

    @property
    def input_variables(self) -> List[str]:
        # see https://pryp.in/blog/16/extract-keys-from-pythons-format-strings.html
        return [t[1] for t in Formatter().parse(self.prompt_template) if t[1] is not None]


# (prompt style,...) -> prompt template
_PROMPT_TEMPLATE_MAPPING = {
    ("default",): PromptInfo(
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    ),
    ("dolly",): PromptInfo(
        prompt_template=DOLLY_PROMPT_TEMPLATE,
    ),
    ("llama", "mistral"): PromptInfo(
        prompt_template=LLAMA_PROMPT_TEMPLATE,
        human_key="",
        ai_key="",
        human_message_prefix="[INST]",
        human_message_suffix="[/INST]",
        ai_message_suffix="</s>"
    ),
    ("oasst", "llama_orca"): PromptInfo(
        prompt_template=OASST_PROMPT_TEMPLATE,
        human_key="",
        ai_key="",
        human_message_prefix="<|prompter|>",
        human_message_suffix="</s>",
        ai_message_prefix="<|assistant|>",
        ai_message_suffix="</s>"
    ),
    ("zephyr",): PromptInfo(
        prompt_template=ZEPHYR_PROMPT_TEMPLATE,
        human_key="",
        ai_key="",
        human_message_prefix="<|user|>",
        human_message_suffix="</s>",
        ai_message_prefix="<|assistant|>",
        ai_message_suffix="</s>"
    ),
}
# normalize/expand the mapping, convert ( (prompt style,...) -> prompt template )
# to ( prompt style -> prompt template )
PROMPT_TEMPLATE_MAPPING = {
    k: v for ks, v in _PROMPT_TEMPLATE_MAPPING.items() for k in ks
}
PROMPT_TEMPLATE_NAMES = [
    "auto", *list(PROMPT_TEMPLATE_MAPPING.keys())
]

# model architecture -> prompt template
ARCHITECTURE_TO_PROMPT_TEMPLATE = {
    "GPTNeoXForCausalLM": "dolly",
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
}
