import time

import requests

from fedml_core.distributed.communication import Observer
from fedml_core.distributed.communication.mqtt import MqttClient


def register():
    str_device_UUID = "klzjiugy9018klskldg109oijkldjf"
    URL = "http://127.0.0.1:5000/api/register"

    newHeaders = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'device_id': str_device_UUID}

    # sending get request and saving the response as response object
    r = requests.post(url=URL, params=PARAMS, headers=newHeaders)
    result = r.json()
    print(result)


def get_training_task_info():
    pass


if __name__ == '__main__':
    register()

    HOST = "81.71.1.31"
    PORT = 1883
    client = MqttClient(HOST, PORT, "TrainingExecutor")


    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))

    client.add_observer(Obs())
    time.sleep(10)
    print('client ID:%s' % client.client_id)
    client.send("hello", "Hello world!")
    client.send("temperature", "266.0")
    client.send("humidity", "65%")
    time.sleep(2)
    client.send("temperature", "15.0")
    print("client, send Fin...")
    time.sleep(10)