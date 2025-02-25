from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
import uuid


class Bot(FedMLPredictor):  # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()

        # --- Your model initialization code here, here is a example ---
        self.uuid = uuid.uuid4()
        # -------------------------------------------
    
    def predict(self, request: dict):
        input_dict = request
        question: str = input_dict.get("text", "").strip()

        # --- Your model inference code here ---
        response = f"I am a replica, my id is {self.uuid}"
        # ---------------------------------------

        return {"v1_generated_text": f"V1: The answer to your question {question} is: {response}"}


if __name__ == "__main__":
    chatbot = Bot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
