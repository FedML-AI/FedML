package ai.fedml.edge.service;

import android.os.Handler;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.nativemobilenn.NativeFedMLClientManager;
import ai.fedml.edge.nativemobilenn.TrainingCallback;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.service.communicator.OnTrainCompletedListener;
import ai.fedml.edge.service.entity.TrainProgress;
import ai.fedml.edge.service.entity.TrainingParams;
import ai.fedml.edge.utils.BackgroundHandler;
import ai.fedml.edge.utils.FileUtils;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;
import androidx.annotation.NonNull;

public class TrainingExecutor implements MessageDefine {
    private static final Handler mBgHandler = new BackgroundHandler("TrainingExecutor");
    private OnTrainProgressListener mOnTrainProgressListener = null;
    private Runnable currentRunnable;
    private final Map<String, Boolean> runStateMap;

    private NativeFedMLClientManager mNativeFedMLClientManager = null;

    public TrainingExecutor(@NonNull final OnTrainProgressListener onTrainProgressListener) {
        mOnTrainProgressListener = onTrainProgressListener;
        runStateMap = new ConcurrentHashMap<>();
    }

    public TrainProgress getTrainProgress() {
        if (mNativeFedMLClientManager != null) {
            String strEpochAndLoss = mNativeFedMLClientManager.getEpochAndLoss();
            String[] arr = strEpochAndLoss.split(",");
            return TrainProgress.builder()
                    .epoch(Integer.parseInt(arr[0]))
                    .loss(Float.parseFloat(arr[1])).build();
        } else {
            return null;
        }
    }

    public void training(final TrainingParams params) throws IOException {
        final long edgeId = params.getEdgeId();
        final long runId = params.getRunId();
        final int clientIdx = params.getClientIdx();
        final int clientRunIdx = params.getClientRound();
        Boolean state = runStateMap.get(runId + "_" + clientRunIdx);
        if (state != null && !state) {
            LogHelper.d("training(%d, %d) stop by user", runId, clientRunIdx);
            return;
        }
        final String dataSet = params.getDataSet();

        // a must check to see whether the dataset path has dataset,
        // otherwise the C++ training engine may not report errors,
        // which would lead to abnormal training accuracy/loss.
        final String trainDataPath = SharePreferencesData.getPrivatePath() + "/" + dataSet;
        if (FileUtils.isEmptyDirectory(trainDataPath)){
            IOException tr = new IOException("The following path does not have dataset for training");
            LogHelper.e(tr, "trainDataPath is empty. Please set the data path correctly: " +
                    trainDataPath + ". Please see the guidance at GitHub FedML/android/data/README.md");
            throw tr;
        }

        final String trainModelPath = params.getTrainModelPath();
        final OnTrainCompletedListener onTrainCompletedListener = params.getListener();
        final int batchSize = params.getBatchSize();
        final double lr = params.getLearningRate();
        final int epochNum = params.getEpochNum();
        final int trainSize = params.getTrainSize();
        final int testSize = params.getTestSize();

        mBgHandler.removeCallbacks(currentRunnable);

        currentRunnable = () -> {
            mNativeFedMLClientManager = new NativeFedMLClientManager();
            LogHelper.d("FedMLDebug. Training Engine Hyperparameters: trainModelPath = %s, " +
                    "trainDataPath = %s, dataSet = %s, trainSize = %d, testSize = %d,\n" +
                    "batchSize = %d, lr = %f, epochNum = %d", trainModelPath,
                    trainDataPath, dataSet, trainSize, testSize,
                    batchSize, lr, epochNum);

            mNativeFedMLClientManager.init(trainModelPath, trainDataPath, dataSet, trainSize, testSize,
                    batchSize, lr, epochNum, new TrainingCallback() {
                        @Override
                        public void onProgress(float progress) {
                            if (mOnTrainProgressListener != null) {
                                mOnTrainProgressListener.onProgressChanged(clientRunIdx, progress);
                            }
                        }

                        @Override
                        public void onAccuracy(int epoch, float accuracy) {
                            if (mOnTrainProgressListener != null) {
                                LogHelper.d("epoch = %d, accuracy = %f", epoch, accuracy);
                                mOnTrainProgressListener.onEpochAccuracy(clientRunIdx, epoch, accuracy);
                            }
                        }

                        @Override
                        public void onLoss(int epoch, float loss) {
                            if (mOnTrainProgressListener != null) {
                                mOnTrainProgressListener.onEpochLoss(clientRunIdx, epoch, loss);
                            }
                        }
                    });
            final String result = mNativeFedMLClientManager.train();
            LogHelper.d("result(%s)", result);
            if (result != null) {
                long trainSamples = Long.parseLong(result);
                LogHelper.d("trainSamples(%d)", trainSamples);
                if (onTrainCompletedListener != null) {
                    onTrainCompletedListener.onTrainCompleted(trainModelPath, edgeId, clientIdx, trainSamples);
                }
            }
        };
        mBgHandler.post(currentRunnable);
    }

    public void resetTrain(final long runId, final int clientRunIndex) {
        runStateMap.put(runId + "_" + clientRunIndex, false);
        mBgHandler.removeCallbacks(currentRunnable);
        if (mNativeFedMLClientManager != null){
            mNativeFedMLClientManager.stopTraining();
            mNativeFedMLClientManager = null;
            LogHelper.d("FedMLDebug. mNativeFedMLClientManager is released.");
        }
    }

    public void stopTrain() {
        runStateMap.clear();
        mBgHandler.removeCallbacks(currentRunnable);
        if (mNativeFedMLClientManager != null){
            mNativeFedMLClientManager.stopTraining();
            mNativeFedMLClientManager = null;
            LogHelper.d("FedMLDebug. mNativeFedMLClientManager is released.");
        }
    }
}
