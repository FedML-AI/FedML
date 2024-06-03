from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
import uuid
import torch

# Calculate the number of elements
num_elements = 1_073_741_824 // 4  # using integer division for whole elements


class DummyPredictor(FedMLPredictor):
    def __init__(self):
        super().__init__()
        # Create a tensor with these many elements
        tensor = torch.empty(num_elements, dtype=torch.float32)

        # Move the tensor to GPU
        tensor_gpu = tensor.cuda()

        # for debug
        with open("/tmp/dummy_gpu_occupier.txt", "w") as f:
            f.write("GPU is occupied")

        self.worker_id = uuid.uuid4()

    def predict(self, request):
        return {f"AlohaV0From{self.worker_id}": request}


if __name__ == "__main__":
    predictor = DummyPredictor()
    fedml_inference_runner = FedMLInferenceRunner(predictor)
    fedml_inference_runner.run()
