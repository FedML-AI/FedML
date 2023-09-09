import os
from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner

# Define the local data cache directory for model and config files
DATA_CACHE_DIR = "~/fedml_serving/model_and_config"
DATA_CACHE_DIR = os.path.expanduser(DATA_CACHE_DIR)  # Use an absolute path

class Chatbot(FedMLPredictor):
    """
    A chatbot powered by language models for generating text-based responses.

    This chatbot uses Hugging Face Transformers to generate text-based responses to user inputs.

    Attributes:
        chatbot (LLMChain): The language model-based chatbot.

    Methods:
        predict(request: dict) -> dict:
            Generate a response to a user's input text.

    Example:
        chatbot = Chatbot()
        fedml_inference_runner = FedMLInferenceRunner(chatbot)
        fedml_inference_runner.run()
    """

    def __init__(self):
        """
        Initialize the Chatbot with a language model-based chatbot.

        This constructor initializes the chatbot by loading a pre-trained language model
        and setting up the necessary components for text generation.
        """
        super().__init__()  # Executes the bootstrap shell script
        from langchain import PromptTemplate, LLMChain
        from langchain.llms import HuggingFacePipeline
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            TextGenerationPipeline,
        )

        PROMPT_FOR_GENERATION_FORMAT = """
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."

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
            torch_dtype=torch.float32,  # float 16 not supported on CPU
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

    def predict(self, request: dict) -> dict:
        """
        Generate a response to a user's input text.

        Args:
            request (dict): A dictionary containing user input text.

        Returns:
            dict: A dictionary containing the generated text-based response.

        Example:
            input_request = {"text": "Tell me a joke."}
            response = chatbot.predict(input_request)
        """
        input_dict = request
        question: str = input_dict.get("text", "").strip()

        if len(question) == 0:
            response_text = "<received empty input; no response generated.>"
        else:
            response_text = self.chatbot.predict(instruction=question)

        return {"generated_text": str(response_text)}

if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
