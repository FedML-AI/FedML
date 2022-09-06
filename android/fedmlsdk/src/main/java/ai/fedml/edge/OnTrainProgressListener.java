package ai.fedml.edge;

public interface OnTrainProgressListener {
    void onEpochLoss(int round, int epoch, float loss);

    void onEpochAccuracy(int round, int epoch, float accuracy);

    void onProgressChanged(int round, float progress);
}
