# Install FedML
```
pip install fedml
```

Note that the mobile platform only supports Python 3.7 due to the contraints of MNN library.

# Start the server for FL in the mobile setting

1. adb push the data to your Android device
```
brew install android-platform-tools
../prepare.sh
```

2. Launch Android Device, and bind the Android Device to open.fedml.ai.

3. check the device ID at open.fedml.ai, and 
   change the edge ID at the test scripts at `test/android_protocol_test/test_protocol.py`
   
   For example, in the following test function, the edge ID list is `[22, 126]`.
```python
    def test_start_train_phone_1(self):
        start_train_json = {"groupid": "38", "clientLearningRate": 0.001, "partitionMethod": "homo",
                            "starttime": 1646068794775, "trainBatchSize": 64, "edgeids": [22, 126],
                            "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6NjAsImFjY291bnQiOiJhbGV4LmxpYW5nIiwibG9naW5UaW1lIjoiMTY0NjA2NTY5MDAwNSIsImV4cCI6MH0.0OTXuMTfxqf2duhkBG1CQDj1UVgconnoSH0PASAEzM4",
                            "modelName": "lenet_mnist",
                            "urls": ["https://fedmls3.s3.amazonaws.com/025c28be-b464-457a-ab17-851ae60767a9"],
                            "clientOptimizer": "adam", "userids": ["60"], "clientNumPerRound": 3, "name": "1646068810",
                            "commRound": 3, "localEpoch": 1, "runId": 189, "id": 169, "projectid": "56",
                            "dataset": "mnist", "communicationBackend": "MQTT_S3", "timestamp": "1646068794778"}
        self.communicator.send_message("flserver_agent/126/start_train", json.dumps(start_train_json))
```

4. start the python server at 
`python/examples/cross_device/mqtt_s3_fedavg_mnist_lr_example/custum_data_and_model/`
   
```
bash run_server.sh
```

5. simulate the MLOps start_run message at `test/android_protocol_test/test_protocol.py`. Please run 
`test_start_train_phone_1()`, and `test_start_train_phone_2()` for each phone.

To understand the detailed workflow, please check the doc at:
https://fedml-inc.larksuite.com/wiki/wikus48ZH9TvRb4MlBlfg7tdsQb