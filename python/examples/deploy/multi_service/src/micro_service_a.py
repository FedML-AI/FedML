from fastapi import FastAPI, Request
import uvicorn

api = FastAPI()


@api.post("/train")
async def train(request: Request):
    input_json = await request.json()
    return {"train_response": "Train Service Success"}


@api.get("/health")
def get_health():
    return {"success": True}


port = 23456
uvicorn.run(api, host="0.0.0.0", port=port)
