from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
import uuid


class DummyPredictor(FedMLPredictor):
    def __init__(self):
        super().__init__()
        self.worker_id = uuid.uuid4()
        
    def predict(self, request):
        return {f"AlohaV1From{self.worker_id}": request}


if __name__ == "__main__":
    predictor = DummyPredictor()
    fedml_inference_runner = FedMLInferenceRunner(predictor)
    fedml_inference_runner.run()
