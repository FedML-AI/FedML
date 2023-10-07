import time

import fedml

if __name__ == "__main__":
    print("Hi everyone, I am an launch job.")

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

    acc = 0.1
    loss = 2.0
    for iter_count in range(20):
        acc += 0.01
        loss -= 0.02
        fedml.mlops.log_metric({"acc": acc, "loss": loss})
        time.sleep(10)
