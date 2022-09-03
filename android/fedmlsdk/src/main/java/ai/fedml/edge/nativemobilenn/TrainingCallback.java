package ai.fedml.edge.nativemobilenn;

public interface TrainingCallback {
    void onProgress(float progress);

    void onAccuracy(int epoch, float accuracy);

    void onLoss(int epoch, float loss);
}
