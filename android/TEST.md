# Train Flow

## 1. onStartTrain

**Received**
topic: flserver_agent/1/start_train

```json
{
  "groupid": "38",
  "clientLearningRate": 0.001,
  "partitionMethod": "homo",
  "starttime": 1646068794775,
  "trainBatchSize": 64,
  "edgeids": [
    17,
    20,
    18,
    21,
    19
  ],
  "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6NjAsImFjY291bnQiOiJhbGV4LmxpYW5nIiwibG9naW5UaW1lIjoiMTY0NjA2NTY5MDAwNSIsImV4cCI6MH0.0OTXuMTfxqf2duhkBG1CQDj1UVgconnoSH0PASAEzM4",
  "modelName": "resnet56",
  "urls": [
    "https://fedmls3.s3.amazonaws.com/025c28be-b464-457a-ab17-851ae60767a9"
  ],
  "clientOptimizer": "adam",
  "userids": [
    "60"
  ],
  "clientNumPerRound": 3,
  "name": "1646068810",
  "commRound": 3,
  "localEpoch": 1,
  "runId": 168,
  "id": 169,
  "projectid": "56",
  "dataset": "cifar10",
  "communicationBackend": "MQTT_S3",
  "timestamp": "1646068794778"
}
```

**Send**
Topic: fedml_168_1

```json
{
  "client_status": "ONLINE",
  "msg_type": 5,
  "receiver": 0,
  "sender": 1
}
```

## 2. init Config

**Received**
Topic: fedml_168_0_1

```json
{
  "msg_type": 1,
  "sender": 0,
  "receiver": 1,
  "model_params": "fedml_111_0_39d756ca2-1ce1-44bc-b232-59f0ae054f0e",
  "client_idx": "0"
}
```

**Send**
Topic: fedml_168_1

```json
 {
  "client_idx": "0",
  "model_params": "fedml_111_0_39d756ca2-1ce1-44bc-b232-59f0ae054f0e",
  "num_samples": 5,
  "msg_type": 3,
  "receiver": 0,
  "sender": 1
}
```

## 2. Sync Config

**Received**
Topic: fedml_168_1

```json
{
  "msg_type": 2,
  "sender": 0,
  "receiver": 1,
  "model_params": "fedml_111_0_39d756ca2-1ce1-44bc-b232-59f0ae054f0e",
  "client_idx": "0"
}
```

**Send**
Topic: fedml_168_1

```json
{
  "client_idx": "0",
  "model_params": "fedml_111_0_39d756ca2-1ce1-44bc-b232-59f0ae054f0e",
  "num_samples": 5,
  "msg_type": 3,
  "receiver": 0,
  "sender": 1
}
```