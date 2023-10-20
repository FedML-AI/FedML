
## Log Metric APIs

log dictionary of metric data to the MLOps platform (open.fedml.ai)
```
fedml.mlops.log(
    metrics: dict, 
    step: int = None, 
    customized_step_key: str = None, 
    commit: bool = True)

fedml.mlops.log_metric(
    metrics: dict, 
    step: int = None, 
    customized_step_key: str = None, 
    commit: bool = True,
    run_id=None, 
    edge_id=None)
```

| Arguments            | Description                                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
| metrics              | A dictionary object for metrics, e.g., {"accuracy": 0.3, "loss": 2.0}                                           |
| step                 | Set the index for current metric. If this value is none, then step will be the current global step counter.     |
| customized_step_key  | Specify the customized step key, which must be one of the keys in the metrics dictionary.                       |
| commit               | If commit is false, the metrics dictionary will be saved to memory and won't be committed until commit is true. |
| run_id               | Run id for the metric object. Default is none, which will be filled automatically.                              |
| edge_id              | Edge id for current device. Default is none, which will be filled automatically.                                |

### Example

```
    fedml.mlops.log({"ACC": 0.1})
    fedml.mlops.log({"acc": 0.11})
    fedml.mlops.log({"acc": 0.2})
    fedml.mlops.log({"acc": 0.3})

    fedml.mlops.log({"acc": 0.31}, step=1)
    fedml.mlops.log({"acc": 0.32, "x_index": 2}, step=2, customized_step_key="x_index")
    fedml.mlops.log({"loss": 0.33}, customized_step_key="x_index", commit=False)
    fedml.mlops.log({"acc": 0.34}, step=4, customized_step_key="x_index", commit=True)
    
    fedml.mlops.log_metric({"acc": 0.8})
```

## Artifacts APIs

log artifacts to the MLOps platform (open.fedml.ai), such as file, log, model, etc.
```
fedml.mlops.log_artifact(
    artifact: Artifact, 
    version=None, 
    run_id=None, 
    edge_id=None)
```

| Arguments        | Description                                                                            |
|------------------|----------------------------------------------------------------------------------------|
| artifact         | An artifact object, e.g., file, log, model, etc.                                       |
| version          | The version of MLOps, options: dev, test, release. Default is release (open.fedml.ai). |
| run_id           | Run id for the artifact object. Default is none, which will be filled automatically.   |
| edge_id          | Edge id for current device. Default is none, which will be filled automatically.       |

### Example 

```
    artifact = fedml.mlops.Artifact(name="general-file", type=fedml.mlops.ARTIFACT_TYPE_NAME_GENERAL)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)

    fedml.mlops.log_model("cv-model", "./requirements.txt")

    artifact = fedml.mlops.Artifact(name="log-file", type=fedml.mlops.ARTIFACT_TYPE_NAME_LOG)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)

    artifact = fedml.mlops.Artifact(name="source-file", type=fedml.mlops.ARTIFACT_TYPE_NAME_SOURCE)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)

    artifact = fedml.mlops.Artifact(name="dataset-file", type=fedml.mlops.ARTIFACT_TYPE_NAME_DATASET)
    artifact.add_file("./requirements.txt")
    artifact.add_dir("./config")
    fedml.mlops.log_artifact(artifact)
```


