from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
from model.mnist_model import LogisticRegression

# This is the model file that will upload to MLOps
# The path is related to the workspace directory
MODEL_PARMS_DIR = "./model/model_parms_from_mlops"
# If you do not want to upload the model file to MLOps,
# (i.e., you want to use the model file in the lcoal DATA_CACHE_DIR)
# Please use the DATA_CACHE_DIR and specify DATA_CACHE_DIR
# in the fedml_model_config.yaml
# DATA_CACHE_DIR = ""

class MnistPredictor(FedMLPredictor):
    def __init__(self):
        import pickle
        import torch

        with open(MODEL_PARMS_DIR, 'rb') as model_file_obj:
            model_params = pickle.load(model_file_obj)
        
        output_dim = 10

        self.model = LogisticRegression(28 * 28, output_dim)

        self.model.load_state_dict(model_params)

        self.list_to_tensor_func = torch.tensor
        
    def predict(self, request):
        arr = request["arr"]
        input_tensor = self.list_to_tensor_func(arr)
        prediction = self.model(input_tensor).tolist()
        return prediction

if __name__ == "__main__":
    predictor = MnistPredictor()
    fedml_inference_runner = FedMLInferenceRunner(predictor)
    fedml_inference_runner.run()