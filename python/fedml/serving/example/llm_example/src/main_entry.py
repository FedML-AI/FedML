from fedml_lib.fedml_serve import FedMLClientPredictor     # TODO: Merge this into real fedml lib
from fedml_lib.fedml_serve import FedMLInferenceRunner

# DATA_CACHE_DIR is a LOCAL folder that contains the model and config files
# If you do not want to transfer the model and config files to MLOps
# Then also metion DATA_CACHE_DIR in the fedml_model_config.yaml
DATA_CACHE_DIR = "~/fedml_serving/model_and_config"

class Chatbot(FedMLClientPredictor):                # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()                          # Will excecute the bootstrap shell script
        from langchain import PromptTemplate, LLMChain
        from langchain.llms import HuggingFacePipeline
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            TextGenerationPipeline,
        )

        PROMPT_FOR_GENERATION_FORMAT = f""""Below is an instruction that describes a task. Write a response that appropriately completes the request."

        ### Instruction:
        {{instruction}}

        ### Response:
        """

        prompt = PromptTemplate(
            input_variables=["instruction"],
            template=PROMPT_FOR_GENERATION_FORMAT
        )

        config = AutoConfig.from_pretrained(DATA_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            DATA_CACHE_DIR,
            torch_dtype=torch.float32,      # float 16 not supported on CPU
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(DATA_CACHE_DIR, device_map="auto")

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
    
    def predict(self, request:dict):
        input_dict = request
        question: str = input_dict.get("text", "").strip()

        if len(question) == 0:
            response_text = "<received empty input; no response generated.>"
        else:
            response_text = self.chatbot.predict(instruction=question)

        return {"generated_text": str(response_text)}

if __name__ == "__main__":
    chatbot_obj = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot_obj)
    fedml_inference_runner.run()