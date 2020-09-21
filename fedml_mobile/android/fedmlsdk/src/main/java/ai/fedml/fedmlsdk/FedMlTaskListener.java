package ai.fedml.fedmlsdk;

import ai.fedml.fedmlsdk.trainingexecutor.ExecutorResponse.TrainingTaskParam;

public interface FedMlTaskListener {
    void onReceive(TrainingTaskParam param);
}
