from dataclasses import asdict
from pprint import pformat
from typing import Dict, List, Optional, Union

from fedml.serving import FedMLPredictor
from jinja2.exceptions import TemplateError

from .model_builder import ModelArguments
from .typing import TokenizerType


class BaseChatbot(FedMLPredictor):
    support_async_predict: bool = False

    def __init__(self, model_args: Optional[ModelArguments] = None):
        super().__init__()

        if model_args is None:
            model_args = ModelArguments.from_environ(
                default_kwargs=dict(
                    model_name_or_path="~/fedml_serving/model_and_config",
                    verbose=True
                )
            )
        self.model_args = model_args

        if self.model_args.verbose:
            print(f"\nmodel config: {pformat(asdict(self.model_args))}", flush=True)

        # private variables
        self._tokenizer = None

    @property
    def tokenizer(self) -> TokenizerType:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: TokenizerType) -> None:
        self._tokenizer = tokenizer

    def apply_chat_template(
            self,
            messages: List[Dict[str, str]],
            tokenize: bool = False,
            add_generation_prompt: bool = True,
            **kwargs
    ) -> Union[str, List[int]]:
        # see https://huggingface.co/docs/transformers/v4.34.1/chat_templating
        try:
            input_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs
            )
        except TemplateError as e:
            # see https://github.com/mistralai/mistral-src/issues/35
            if messages[0]["role"] == "system":
                # mistral models requires format
                #   <s>[INST] System Prompt + Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
                #   need to combine system prompt with first user message or convert it to user message
                messages[0]["role"] = "user"
                if len(messages) > 1 and messages[1]["role"] == "user":
                    messages[1]["content"] = f"{messages[0]['content']}\n{messages[1]['content']}"
                    del messages[0]

                input_str = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs
                )
            else:
                raise e

        return input_str
