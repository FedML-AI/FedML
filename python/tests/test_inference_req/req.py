import requests

url = ""
model_name = ""
endpoint_name = ""
token = ""
inputs = {"text": "What is a good cure for hiccups?"}

data = {"model_name": model_name, "inputs": inputs, "end_point_name": endpoint_name, "token": token}

response = requests.post(url, json=data)
print(response.json())
