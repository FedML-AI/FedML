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
        question: str = input_dict.get("text", "").strip()

        if len(question) == 0:
            response_text = "<received empty input; no response generated.>"
        else:
            response_text = self.chatbot.predict(instruction=question)

        try:
            unique_id = "IMAGE_MODEL_KEY"
            with open(f"{unique_id}.txt", "w") as f:
                f.write(question)
                f.write("\n\n")
                f.write(response_text)
                f.write("\n\n")
            response = fedml.api.upload(data_path=f"{unique_id}.txt", name=unique_id,
                                        api_key=os.environ.get("NEXUS_API_KEY", None), metadata={"type": "chatbot"})
            print(f"upload response: code {response.code}, message {response.message}, data {response.data}")
        except Exception as e:
            pass

        return {"response": str(response_text)}


if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
