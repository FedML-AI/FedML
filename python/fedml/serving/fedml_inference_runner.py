from abc import ABC
from fastapi import FastAPI, Request

class FedMLInferenceRunner(ABC):
    def __init__(self, client_predictor):
        self.client_predictor = client_predictor
    
    def run(self):
        api = FastAPI()

        @api.post("/predict")
        async def predict(request: Request):
            input_json = await request.json()
            response_text = self.client_predictor.predict(input_json)

            return {"generated_text": str(response_text)}
        
        @api.get("/ready")
        async def ready():
            return {"status": "Success"}
        
        import uvicorn
        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)
