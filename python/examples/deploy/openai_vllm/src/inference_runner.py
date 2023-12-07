from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fedml.serving import FedMLInferenceRunner

from .base_chatbot import BaseChatbot


class ChatbotInferenceRunner(FedMLInferenceRunner):
    def __init__(self, client_predictor: BaseChatbot):
        super().__init__(client_predictor)

    def run(self):
        api = FastAPI()

        @api.post("/predict")
        async def predict(request: Request):
            input_json = await request.json()

            if self.client_predictor.support_async_predict:
                return StreamingResponse(self.client_predictor.predict(input_json), media_type="application/json")

            else:
                response_text = self.client_predictor.predict(input_json)

                return {"generated_text": str(response_text)}

        @api.get("/ready")
        async def ready():
            return {"status": "Success"}

        import uvicorn
        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)
