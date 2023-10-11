from abc import ABC
from fastapi import FastAPI, Request

class FedMLInferenceRunner(ABC):
    """
    Abstract base class for federated machine learning inference runners.

    Subclasses should implement the `predict` method for making predictions.

    Attributes:
        client_predictor: An instance of a client predictor class that implements the `predict` method.

    Methods:
        run(): Start the FastAPI server to handle prediction requests.
    """

    def __init__(self, client_predictor):
        """
        Initializes the FedMLInferenceRunner.

        Args:
            client_predictor: An instance of a client predictor class that implements the `predict` method.
        """
        self.client_predictor = client_predictor
    
    def run(self):
        """
        Start the FastAPI server to handle prediction requests.

        This method creates an HTTP server using FastAPI and defines two routes: '/predict' for making predictions
        and '/ready' to check the server's readiness.

        Returns:
            None
        """
        api = FastAPI()
        
        @api.post("/predict")
        async def predict(request: Request):
            """
            Handle POST requests to the '/predict' route for making predictions.

            Args:
                request: The HTTP request object containing the input data.

            Returns:
                dict: A JSON response containing the generated text.
            """
            input_json = await request.json()
            response_text = self.client_predictor.predict(input_json)
            
            return {"generated_text": str(response_text)}
        
        @api.get("/ready")
        async def ready():
            """
            Handle GET requests to the '/ready' route to check the server's readiness.

            Returns:
                dict: A JSON response indicating the server's readiness status.
            """
            return {"status": "Success"}
        
        import uvicorn
        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)
