from abc import ABC
import asyncio

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, StreamingResponse


class FedMLInferenceRunner(ABC):
    def __init__(self, client_predictor):
        self.client_predictor = client_predictor

    def run(self):
        api = FastAPI()

        @api.post("/predict")
        async def predict(request: Request):
            input_json = await request.json()
            header = request.headers.get("Accept", "application/json")
            if header == "application/json" or header == "*/*":
                if input_json.get("stream", False):
                    resp = self.client_predictor.async_predict(input_json)
                    if asyncio.iscoroutine(resp):
                        resp = await resp

                    if isinstance(resp, Response):
                        return resp
                    else:
                        return StreamingResponse(resp)
                else:
                    return self.client_predictor.predict(input_json)
            else:
                response_obj = self.client_predictor.predict(input_json, request.headers.get("Accept"))
                return FileResponse(response_obj)

        @api.get("/ready")
        async def ready():
            return {"status": "Success"}

        import uvicorn
        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)
