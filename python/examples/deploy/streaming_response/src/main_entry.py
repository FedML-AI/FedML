from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
import time
class Chatbot(FedMLPredictor):                # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()
    
    async def async_predict(self, *args):
        input_json = args[0]
        question = input_json.get("text", "[Empty question]")
        for i in range(5):
            yield f"Answer for {question} is: {i+1}\n\n"
            time.sleep(1)

if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()