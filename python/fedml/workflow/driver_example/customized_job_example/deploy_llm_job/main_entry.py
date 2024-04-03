import os

import fedml.api
from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
)
import uuid


class Chatbot(FedMLPredictor):  # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()
        PROMPT_FOR_GENERATION_FORMAT = f""""Below is an instruction that describes a task. Write a response that appropriately completes the request."

        ### Instruction:
        {{instruction}}

        ### Response:
        """

        prompt = PromptTemplate(
            input_variables=["instruction"],
            template=PROMPT_FOR_GENERATION_FORMAT
        )

        config = AutoConfig.from_pretrained("EleutherAI/pythia-70m")
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m",
            torch_dtype=torch.float32,  # float 16 not supported on CPU
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", device_map="auto")

        hf_pipeline = HuggingFacePipeline(
            pipeline=TextGenerationPipeline(
                model=model,
                tokenizer=tokenizer,
                return_full_text=True,
                task="text-generation",
                do_sample=True,
                max_new_tokens=256,
                top_p=0.92,
                top_k=0
            )
        )
        self.chatbot = LLMChain(llm=hf_pipeline, prompt=prompt, verbose=True)

    def predict(self, request: dict):
        input_dict = request

        # If the ouptput of previous job is present, then use this output value to predict.
        # Here inference_job_0 is the name of prevous job.
        # You may use this method to get outputs of all previous jobs
        output_of_previous_job = input_dict.get("inference_job_0")
        if output_of_previous_job is not None:
            question: str = output_of_previous_job
        else:
            question: str = input_dict.get("text", "").strip()

        if len(question) == 0:
            response_text = "<received empty input; no response generated.>"
        else:
            response_text = self.chatbot.predict(instruction=question)

        return {"response": str(response_text)}


if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
