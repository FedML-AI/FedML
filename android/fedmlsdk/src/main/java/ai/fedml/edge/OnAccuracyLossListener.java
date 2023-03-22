package ai.fedml.edge;

public interface OnAccuracyLossListener {
    void onEpochLoss(int epoch, float loss);

    void onEpochAccuracy(int epoch, float accuracy);

    void onProgressChanged(int progress);
}
