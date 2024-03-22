import asyncio
import logging
import time
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse
from fedml.serving import FedMLInferenceRunner, FedMLPredictor


class Chatbot(FedMLPredictor):  # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()

    def predict(self, *args, **kwargs):
        return {"my_output": "example output"}

    async def async_predict(self, *args):
        await asyncio.sleep(0.2)
        return await self._async_predict(*args)

    async def _async_predict(self, *args):
        # time.sleep(0.6)             # Some blocking operation
        await asyncio.sleep(0.3)    # Some async operation
        return 1


if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run(timeout=0.4)
