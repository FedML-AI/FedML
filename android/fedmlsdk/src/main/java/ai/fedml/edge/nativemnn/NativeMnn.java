package ai.fedml.edge.nativemnn;

import android.util.Log;

import ai.fedml.edge.OnAccuracyLossListener;

public class NativeMnn {
    public static final String TAG = "MNNDataNative";

    // load libraries
    static void loadGpuLibrary(String name) {
        try {
            System.loadLibrary(name);
        } catch (Throwable ce) {
            Log.w(TAG, "load MNNTrain " + name + " GPU so exception=%s", ce);
        }
    }

    // load mnn library
    static {
        loadGpuLibrary("MNNTrain");
        loadGpuLibrary("MNN_Express");
        loadGpuLibrary("MNN");
        loadGpuLibrary("MobileNN");
    }

    /**
     * We call the training module of C++ to complete training task, and obtain the relevant information required by Android UI
     *
     * @param modelCachePath the storage path of model in Android
     * @param dataCachePath  the storage path of training dataset in Android
     * @param dataSet        dataset type 0: mnist dataset, 1: cifar10 dataset. Dataset type should be matched with dataset
     * @param batchSizeNum   batch size
     * @param epochNum       the number of epoch
     * @param LearningRate   learning rate
     * @param listener       OnEpochLossListener
     * @return result of client training with total local epochs (format: "loss,trainSamples,accuracy,testSamples")
     */
    public static native String train(String modelCachePath, String dataCachePath, String dataSet,
                                      int trainSize, int testSize,
                                      int batchSizeNum, double LearningRate, int epochNum,
                                      OnAccuracyLossListener listener);

    /**
     * the local epoch index in each global epoch training, and the training loss in this local epoch
     *
     * @return current epoch and the loss value in this epoch (format: "epoch,loss")
     */
    public static native String getEpochAndLoss();

    /**
     * Stop the current training
     *
     * @return success
     */
    public static native boolean stopTraining();

}
