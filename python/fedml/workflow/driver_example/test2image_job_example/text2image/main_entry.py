from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner


class Chatbot(FedMLPredictor):  # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()

    def predict(self, request: dict):
        print(request)

        return {"response": "success"}


if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
