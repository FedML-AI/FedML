from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
import requests


class Chatbot(FedMLPredictor):                # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()
    
    def predict(self, request: dict):
        input_dict = request
        question: str = input_dict.get("text", "").strip()

        if len(question) == 0:
            response_text = "<received empty input; no response generated.>"
        else:
            # Redirect the input to microservice A
            response = requests.post("http://localhost:23456/train", json=input_dict)
            response_text = response.json()["train_response"]

        return {"generated_text": str(response_text)}

    def ready(self) -> bool:
        print("Overwrite the original ready method!")

        response = None
        try:
            response = requests.get("http://localhost:23456/health")
        except:
            pass

        if not response or response.status_code != 200:
            return False
        return True


if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
