package ai.fedml.edge.nativemobilenn;

import android.util.Log;

public final class NativeFedMLClientManager {
    // load libraries
    static void loadGpuLibrary(String name) {
        try {
            System.loadLibrary(name);
        } catch (Throwable ce) {
            Log.w("NativeFedMLUniTrainer", "load MNNTrain " + name + " GPU so exception.", ce);
        }
    }

    // load mnn library
    static {
        loadGpuLibrary("MNNTrain");
        loadGpuLibrary("MNN_Express");
        loadGpuLibrary("MNN");
        loadGpuLibrary("MobileNN");
    }

    private final long mTrainerPtr;
    private TrainingCallback mTrainingCallback;

    public NativeFedMLClientManager() {
        mTrainerPtr = create();
    }

    @Override
    protected void finalize() throws Throwable {
        // call automatically
        release(mTrainerPtr);
    }

    public void init(final String modelCachePath, final String dataCachePath, final String dataSet,
                     int trainSize, int testSize, int batchSizeNum, double learningRate, int epochNum,
                     TrainingCallback trainingCallback) {
        mTrainingCallback = trainingCallback;
        init(mTrainerPtr, modelCachePath, dataCachePath, dataSet,
                trainSize, testSize, batchSizeNum, learningRate, epochNum,
                trainingCallback);
    }

    /**
     * start training
     *
     * @return status
     */
    public String train() {
        return train(mTrainerPtr);
    }

    /**
     * the local epoch index in each global epoch training, and the training loss in this local epoch
     *
     * @return current epoch and the loss value in this epoch (format: "epoch,loss")
     */
    public String getEpochAndLoss() {
        return getEpochAndLoss(mTrainerPtr);
    }

    /**
     * Stop the current training
     *
     * @return success
     */
    public boolean stopTraining() {
        return stopTraining(mTrainerPtr);
    }

    private native long create();

    private native void release(final long pointer);

    private native void init(final long pointer, final String modelCachePath, final String dataCachePath, final String dataSet,
                             int trainSize, int testSize, int batchSizeNum, double learningRate, int epochNum,
                             TrainingCallback trainingCallback);

    private native String train(final long pointer);

    private native String getEpochAndLoss(final long pointer);

    private native boolean stopTraining(final long pointer);
}
