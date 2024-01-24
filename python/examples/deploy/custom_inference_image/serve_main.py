from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner


class DummyPredictor(FedMLPredictor):
    def __init__(self):
        super().__init__()
        
    def predict(self, request):
        return {"Aloha": request}


if __name__ == "__main__":
    predictor = DummyPredictor()
    fedml_inference_runner = FedMLInferenceRunner(predictor)
    fedml_inference_runner.run()