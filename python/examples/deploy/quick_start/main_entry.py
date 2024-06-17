from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner


class Bot(FedMLPredictor):  # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()

        # --- Your model initialization code here ---

        # -------------------------------------------
    
    def predict(self, request: dict):
        input_dict = request
        question: str = input_dict.get("text", "").strip()

        # --- Your model inference code here ---
        response = "I do not know the answer to your question."
        # ---------------------------------------

        return {"generated_text": f"The answer to your question {question} is: {response}"}


if __name__ == "__main__":
    chatbot = Bot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
