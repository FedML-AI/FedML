import requests


def register():
    str_device_UUID = "klzjiugy9018klskldg109oijkldjf"
    URL = "http://127.0.0.1:5000/api/register"

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'device_id': str_device_UUID}

    # sending get request and saving the response as response object
    r = requests.post(url=URL, params=PARAMS)
    result = r.json()
    print(result)


def get_training_task_info():
    pass


if __name__ == '__main__':
    register()
