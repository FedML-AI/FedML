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
    """
    A custom predictor for MNIST digit classification using a logistic regression model.

    This class loads a pretrained logistic regression model and provides a predict method to make predictions
    on input data.

    Args:
        None

    Example:
        predictor = MnistPredictor()
        input_data = {"arr": [0.1, 0.2, 0.3, ..., 0.9]}
        prediction = predictor.predict(input_data)
    """
    def __init__(self):
        """
        Initialize the MnistPredictor by loading a pretrained logistic regression model.
        """
        import pickle
        import torch

        with open(MODEL_PARMS_DIR, 'rb') as model_file_obj:
            model_params = pickle.load(model_file_obj)
        
        output_dim = 10

        self.model = LogisticRegression(28 * 28, output_dim)

        self.model.load_state_dict(model_params)

        self.list_to_tensor_func = torch.tensor
        
    def predict(self, request):
        """
        Perform predictions on input data using the pretrained logistic regression model.

        Args:
            request (dict): A dictionary containing input data for prediction.
                The dictionary should have the following key:
                - "arr" (list): A list of float values representing the input features for a MNIST digit image.

        Returns:
            torch.Tensor: A tensor representing the model's prediction.

        Example:
            predictor = MnistPredictor()
            input_data = {"arr": [0.1, 0.2, 0.3, ..., 0.9]}
            prediction = predictor.predict(input_data)

        Note:
            The input data should be a list of float values with the same dimensionality as the model's input.
        """
        arr = request["arr"]
        input_tensor = self.list_to_tensor_func(arr)
        return self.model(input_tensor)

if __name__ == "__main__":
    predictor = MnistPredictor()
    fedml_inference_runner = FedMLInferenceRunner(predictor)
    fedml_inference_runner.run()