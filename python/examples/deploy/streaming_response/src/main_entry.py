import asyncio
from typing import AsyncGenerator

from fedml.serving import FedMLInferenceRunner, FedMLPredictor


class Chatbot(FedMLPredictor):  # Inherit FedMLClientPredictor
    def __init__(self):
        super().__init__()

    async def async_predict(self, *args) -> AsyncGenerator[str, None]:
        # This function can also return fastapi.responses.StreamingResponse directly
        input_json = args[0]
        question = input_json.get("text", "[Empty question]")
        for i in range(5):
            yield f"Answer for {question} is: {i + 1}\n\n"
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
