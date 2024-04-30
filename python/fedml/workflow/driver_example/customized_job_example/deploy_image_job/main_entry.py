from fedml.serving import FedMLPredictor
from fedml.serving import FedMLInferenceRunner
from model.minist_model import LogisticRegression

# This is the model file that will upload to MLOps
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
        input_dict = request
        arr = request["arr"]

        # If the output of previous job is present, then use this output value to predict.
        # Here inference_job_0 is the name of previous job.
        # You may use this method to get outputs of all previous jobs
        output_of_previous_job = input_dict.get("inference_job_0")
        if output_of_previous_job is not None:
            question: str = output_of_previous_job
        else:
            question: str = input_dict.get("text", "").strip()

        input_tensor = self.list_to_tensor_func(arr)
        return self.model(input_tensor)


if __name__ == "__main__":
    predictor = MnistPredictor()
    fedml_inference_runner = FedMLInferenceRunner(predictor)
    fedml_inference_runner.run()
