from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
from engine_downloader import download_engine
import run_utils
import os


class ScaleLLMBot(FedMLPredictor):
    def __init__(self):
        super().__init__()

        self.args = run_utils.parse_arguments()

        self.args.tokenizer_dir = "meta-llama/Llama-2-13b-chat-hf"
        engine_name = os.environ.get("PREBUILT_ENGINE", "MythoMax-L2-13b")
        self.args.engine_dir = download_engine(engine_name)

        self.decoder, self.tokenizer, self.model_config, self.sampling_config, self.runtime_rank = \
            run_utils.init_bot(**vars(self.args))
    
    def predict(self, request:dict):
        input_dict = request
        question: str = input_dict.get("text", "").strip()

        if len(question) == 0:
            response_text = "<received empty input; no response generated.>"
        else:
            response_text = run_utils.predict(
                self.decoder,
                self.tokenizer,
                self.model_config,
                self.sampling_config,
                self.runtime_rank,
                input_text=question,
                **vars(self.args)
            )

        return {"generated_text": str(response_text)}


if __name__ == "__main__":
    chatbot = ScaleLLMBot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
