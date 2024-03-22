import logging
from abc import ABC
import asyncio

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse


class FedMLInferenceRunner(ABC):
    def __init__(self, client_predictor):
        self.client_predictor = client_predictor
        self._pending_tasks = deque()

    async def run_pending_tasks(self):
        """
        Place the pending task future in the queue, that will poped up if the gpu is available.
        TODO (Raphael): Support Batch Inference
        """
        pass

    def run(self, timeout=None):
        api = FastAPI()

        @api.post("/predict")
        async def predict(request: Request):
            input_json = await request.json()
            header = request.headers.get("Accept", "application/json")
            if header == "application/json" or header == "*/*":
                if input_json.get("stream", False):
                    if timeout is not None:
                        resp = self.client_predictor.async_predict(input_json)
                        if asyncio.iscoroutine(resp):
                            try:
                                resp = await asyncio.wait_for(resp, timeout)
                            except asyncio.TimeoutError:
                                logging.info("Request timed out")
                                return Response(status_code=status.HTTP_408_REQUEST_TIMEOUT)
                            else:
                                # Return in time
                                pass
                    else:
                        resp = self.client_predictor.async_predict(input_json)
                        if asyncio.iscoroutine(resp):
                            resp = await resp

                    if isinstance(resp, Response):
                        return resp
                    else:
                        # if can be streamed (iterable)
                        if hasattr(resp, "__iter__"):
                            return StreamingResponse(resp)
                        else:
                            return resp
                else:
                    resp = self.client_predictor.predict(input_json)
                    if asyncio.iscoroutine(resp):
                        resp = await resp
                    return resp
            else:
                response_obj = self.client_predictor.predict(input_json, request.headers.get("Accept"))
                return FileResponse(response_obj)

        @api.get("/ready")
        async def ready():
            if self.client_predictor.ready():
                return {"status": "Success"}
            else:
                return Response(status_code=status.HTTP_202_ACCEPTED)

        import uvicorn
        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port, log_level="info")
