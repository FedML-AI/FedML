import logging
from abc import ABC
import time
import asyncio

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse


class FedMLInferenceRunner(ABC):
    def __init__(self, client_predictor):
        self.client_predictor = client_predictor

        # TODO(Raphael): Add NTP to sync container's time with host's time

    def run(self):
        api = FastAPI()

        @api.post("/predict")
        async def predict(request: Request):
            """
            fedml_t3: timestamp the request was received by the replica
            fedml_t4: timestamp the request was processed by the replica and the response was sent
            """
            input_json = await request.json()
            header = request.headers.get("Accept", "application/json")

            t3 = time.time_ns()

            if header == "application/json" or header == "*/*":
                if input_json.get("stream", False):
                    resp = self.client_predictor.async_predict(input_json)
                    if asyncio.iscoroutine(resp):
                        resp = await resp

                    t4 = time.time_ns()

                    if isinstance(resp, Response):
                        resp.headers["fedml_t3"] = str(t3)
                        resp.headers["fedml_t4"] = str(t4)
                        return resp
                    else:
                        return StreamingResponse(resp, headers={"fedml_t3": str(t3), "fedml_t4": str(t4)})
                else:
                    resp = self.client_predictor.predict(input_json)
                    if asyncio.iscoroutine(resp):
                        resp = await resp

                    t4 = time.time_ns()

                    if isinstance(resp, Response):
                        try:
                            resp.headers["fedml_t3"] = str(t3)
                            resp.headers["fedml_t4"] = str(t4)
                        except Exception as e:
                            logging.error(f"Response object does not have headers attribute, type: {type(resp)},"
                                          f" error: {e}")
                    elif isinstance(resp, dict):
                        resp = JSONResponse(resp, headers={"fedml_t3": str(t3), "fedml_t4": str(t4)})
                    return resp
            else:
                response_obj = self.client_predictor.predict(input_json, request.headers.get("Accept"))

                t4 = time.time_ns()

                return FileResponse(response_obj, headers={"fedml_t3": str(t3), "fedml_t4": str(t4)})

        @api.get("/ready")
        async def ready():
            if self.client_predictor.ready():
                return {"status": "Success"}
            else:
                return Response(status_code=status.HTTP_202_ACCEPTED)

        import uvicorn
        port = 2345
        uvicorn.run(api, host="0.0.0.0", port=port)
