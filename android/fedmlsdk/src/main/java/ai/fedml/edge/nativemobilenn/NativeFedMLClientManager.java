package ai.fedml.edge.nativemobilenn;

import ai.fedml.edge.utils.LogHelper;

public final class NativeFedMLClientManager {
    // load libraries
    static void loadGpuLibrary(String name) {
        System.loadLibrary(name);
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
        LogHelper.i("NativeFedMLClientManager FedMLDebug. finalize");
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
