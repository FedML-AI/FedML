package ai.fedml.edge.nativemnn;

import android.util.Log;

public final class NativeFedMLTrainer {
    // load libraries
    static void loadGpuLibrary(String name) {
        try {
            System.loadLibrary(name);
        } catch (Throwable ce) {
            Log.w("NativeFedMLTrainer", "load MNNTrain " + name + " GPU so exception=%s", ce);
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

    public NativeFedMLTrainer() {
        mTrainerPtr = create();
    }

    @Override
    protected void finalize() throws Throwable {
        release(mTrainerPtr);
    }

    public void init(final String modelCachePath, final String dataCachePath, final String dataSet,
                     int trainSize, int testSize, int batchSizeNum, double learningRate, int epochNum,
                     int q_bits, int p, int client_num, TrainingCallback trainingCallback) {
        mTrainingCallback = trainingCallback;
        init(mTrainerPtr, modelCachePath, dataCachePath, dataSet, trainSize, testSize, batchSizeNum, learningRate,
                epochNum, q_bits, p, client_num, trainingCallback);
    }

    /**
     * generate local mask and encode mask to share with other users
     *
     * @return maskMatrix
     */
    public float[][] getLocalEncodedMask() {
        return getLocalEncodedMask(mTrainerPtr);
    }

    /**
     * save mask from paired clients
     *
     * @param client_index      client id
     * @param local_encode_mask mask array
     */
    public void saveMaskFromPairedClients(int client_index, float[] local_encode_mask) {
        saveMaskFromPairedClients(mTrainerPtr, client_index, local_encode_mask);
    }

    /**
     * receive client index from surviving users
     *
     * @return client index array
     */
    public int[] getClientIdsThatHaveSentMask() {
        return getClientIdsThatHaveSentMask(mTrainerPtr);
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
     * get masked model after the local training is done
     * the model file is saved at the original path "modelCachePath"
     */
    public void generateMaskedModel() {
        generateMaskedModel(mTrainerPtr);
    }

    /**
     * the server will ask those clients that are online to send aggregated encoded masks
     *
     * @param survivingListFromServer array
     * @return maskArr
     */
    public float[] getAggregatedEncodedMask(int[] survivingListFromServer) {
        return getAggregatedEncodedMask(mTrainerPtr, survivingListFromServer);
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
                             int q_bits, int p, int client_num, TrainingCallback trainingCallback);

    private native float[][] getLocalEncodedMask(final long pointer);

    private native void saveMaskFromPairedClients(final long pointer, int client_index, float[] local_encode_mask);

    private native int[] getClientIdsThatHaveSentMask(final long pointer);

    private native String train(final long pointer);

    private native void generateMaskedModel(final long pointer);

    private native float[] getAggregatedEncodedMask(final long pointer, int[] survivingListFromServer);

    private native String getEpochAndLoss(final long pointer);

    private native boolean stopTraining(final long pointer);
}
