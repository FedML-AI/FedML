"""
Adapted from https://github.com/databrickslabs/dolly/blob/master/training/consts.py
"""

# -----------------------------------------------------------------
DEFAULT_MAX_SEQ_LENGTH = 1024
IGNORE_INDEX = -100

# -----------------------------------------------------------------
MODEL_NAMES = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/gpt-j-6B",
    "databricks/dolly-v2-3b",
    "databricks/dolly-v2-7b",
    "databricks/dolly-v2-12b",
]

# -----------------------------------------------------------------
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

# This is a training prompt that does not contain an input string. The instruction by itself has enough information
# to respond.For example, the instruction might ask for the year a historic figure was born.
PROMPT_NO_INPUT_FORMAT = f"""{INTRO_BLURB}

{INSTRUCTION_KEY}
{{instruction}}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""

# This is a training prompt that contains an input string that serves as context for the instruction. For example,
# the input might be a passage from Wikipedia and the instruction is to extract some information from it.
PROMPT_WITH_INPUT_FORMAT = f"""{INTRO_BLURB}

{INSTRUCTION_KEY}
{{instruction}}

{INPUT_KEY}
{{input}}

{RESPONSE_KEY}
{{response}}

{END_KEY}"""

# This is the prompt that is used for generating responses using an already trained model. It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = f"""{INTRO_BLURB}

{INSTRUCTION_KEY}
{{instruction}}

{RESPONSE_KEY}
"""