import json
import unittest
import time

import mqtt_communicator


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.communicator = mqtt_communicator.EdgeCommunicator(host="mqtt.fedml.ai")

    def test_something(self):
        self.communicator.send_message(
            "fl_client/mlops/status",
            json.dumps({"edge_id": "687c12fdaf43b758", "status": "IDLE"}),
        )
        time.sleep(10)
        self.assertEqual(True, True)

    def test_start_train_phone_1(self):
        start_train_json = {
            "groupid": "38",
            "clientLearningRate": 0.001,
            "partitionMethod": "homo",
            "starttime": 1646068794775,
            "trainBatchSize": 64,
            "edgeids": [22, 126, 27],
            "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6NjAsImFjY291bnQiOiJhbGV4LmxpYW5nIiwibG9naW5UaW1lIjoiMTY0NjA2NTY5MDAwNSIsImV4cCI6MH0.0OTXuMTfxqf2duhkBG1CQDj1UVgconnoSH0PASAEzM4",
            "modelName": "lenet_mnist",
            "urls": [
                "https://fedmls3.s3.amazonaws.com/025c28be-b464-457a-ab17-851ae60767a9"
            ],
            "clientOptimizer": "adam",
            "userids": ["60"],
            "clientNumPerRound": 3,
            "name": "1646068810",
            "commRound": 3,
            "localEpoch": 1,
            "runId": 189,
            "id": 169,
            "projectid": "56",
            "dataset": "mnist",
            "communicationBackend": "MQTT_S3",
            "timestamp": "1646068794778",
        }
        self.communicator.send_message(
            "flserver_agent/126/start_train", json.dumps(start_train_json)
        )

    def test_start_train_phone_2(self):
        start_train_json = {
            "groupid": "38",
            "clientLearningRate": 0.001,
            "partitionMethod": "homo",
            "starttime": 1646068794775,
            "trainBatchSize": 64,
            "edgeids": [22, 126, 27],
            "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6NjAsImFjY291bnQiOiJhbGV4LmxpYW5nIiwibG9naW5UaW1lIjoiMTY0NjA2NTY5MDAwNSIsImV4cCI6MH0.0OTXuMTfxqf2duhkBG1CQDj1UVgconnoSH0PASAEzM4",
            "modelName": "lenet_mnist",
            "urls": [
                "https://fedmls3.s3.amazonaws.com/025c28be-b464-457a-ab17-851ae60767a9"
            ],
            "clientOptimizer": "adam",
            "userids": ["60"],
            "clientNumPerRound": 3,
            "name": "1646068810",
            "commRound": 3,
            "localEpoch": 1,
            "runId": 189,
            "id": 169,
            "projectid": "56",
            "dataset": "mnist",
            "communicationBackend": "MQTT_S3",
            "timestamp": "1646068794778",
        }
        self.communicator.send_message(
            "flserver_agent/22/start_train", json.dumps(start_train_json)
        )

    def test_start_train_phone_3(self):
        start_train_json = {
            "groupid": "38",
            "clientLearningRate": 0.001,
            "partitionMethod": "homo",
            "starttime": 1646068794775,
            "trainBatchSize": 64,
            "edgeids": [22, 126, 27],
            "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6NjAsImFjY291bnQiOiJhbGV4LmxpYW5nIiwibG9naW5UaW1lIjoiMTY0NjA2NTY5MDAwNSIsImV4cCI6MH0.0OTXuMTfxqf2duhkBG1CQDj1UVgconnoSH0PASAEzM4",
            "modelName": "lenet_mnist",
            "urls": [
                "https://fedmls3.s3.amazonaws.com/025c28be-b464-457a-ab17-851ae60767a9"
            ],
            "clientOptimizer": "adam",
            "userids": ["60"],
            "clientNumPerRound": 3,
            "name": "1646068810",
            "commRound": 3,
            "localEpoch": 1,
            "runId": 189,
            "id": 169,
            "projectid": "56",
            "dataset": "mnist",
            "communicationBackend": "MQTT_S3",
            "timestamp": "1646068794778",
        }
        self.communicator.send_message(
            "flserver_agent/27/start_train", json.dumps(start_train_json)
        )

    def test_init_config(self):
        config_json = {
            "msg_type": 1,
            "sender": 0,
            "receiver": 17,
            "model_params": "fedml_189_0_17_a3759e349d4211ec8f7d60f81da88740",
            "client_idx": "2",
        }
        self.communicator.send_message("fedml_189_0_17", json.dumps(config_json))

    def test_sync_config(self):
        config_json = {
            "msg_type": 1,
            "sender": 0,
            "receiver": 17,
            "model_params": "fedml_111_0_39d756ca2-1ce1-44bc-b232-59f0ae054f0e",
            "client_idx": "0",
        }
        self.communicator.send_message("fedml_189_0_17", json.dumps(config_json))

    def test_send_model(self):
        json_data = {
            "client_idx": "1",
            "model_params": "fedml_196_0_17-98737f1276394c3a8bfa06b45238030e",
            "num_samples": 60032,
            "msg_type": 2,
            "receiver": 0,
            "sender": 17,
        }
        self.communicator.send_message("fedml_189_126", json.dumps(json_data))

    def test_send_model_2(self):
        json_data = {
            "client_idx": "2",
            "model_params": "fedml_189_0_17_a3759e349d4211ec8f7d60f81da88740",
            "num_samples": 60032,
            "msg_type": 3,
            "receiver": 0,
            "sender": 27,
        }
        self.communicator.send_message("fedml_189_27", json.dumps(json_data))


if __name__ == "__main__":
    unittest.main()
