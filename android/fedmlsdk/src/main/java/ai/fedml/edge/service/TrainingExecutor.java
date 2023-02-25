package ai.fedml.edge.service;

import android.os.Handler;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.nativemobilenn.NativeFedMLClientManager;
import ai.fedml.edge.nativemobilenn.TrainingCallback;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.service.component.OnTrainCompletedListener;
import ai.fedml.edge.service.component.MetricsReporter;
import ai.fedml.edge.service.entity.TrainProgress;
import ai.fedml.edge.service.entity.TrainingParams;
import ai.fedml.edge.utils.BackgroundHandler;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;
import androidx.annotation.NonNull;

public class TrainingExecutor implements MessageDefine {
    private static final Handler mBgHandler = new BackgroundHandler("TrainingExecutor");
    private OnTrainProgressListener mOnTrainProgressListener = null;
    private final MetricsReporter mReporter;
    private Runnable currentRunnable;
    private final Map<String, Boolean> runStateMap;

    private NativeFedMLClientManager mNativeFedMLClientManager = null;

    public TrainingExecutor(@NonNull final OnTrainProgressListener onTrainProgressListener) {
        mOnTrainProgressListener = onTrainProgressListener;
        mReporter = MetricsReporter.getInstance();
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

    public void training(final TrainingParams params) {
        final long edgeId = params.getEdgeId();
        final long runId = params.getRunId();
        final int clientIdx = params.getClientIdx();
        final int clientRunIdx = params.getClientRound();
        Boolean state = runStateMap.get(runId + "_" + clientRunIdx);
        if (state != null && !state) {
            LogHelper.d("training(%d, %d) stop by user", runId, clientRunIdx);
            return;
        }
        final String trainDataPath = SharePreferencesData.getPrivatePath();
        final String trainModelPath = params.getTrainModelPath();
        final OnTrainCompletedListener listener = params.getListener();
        final String dataSet = params.getDataSet();
        final int batchSize = params.getBatchSize();
        final double lr = params.getLearningRate();
        final int epochNum = params.getEpochNum();
        final int trainSize = params.getTrainSize();
        final int testSize = params.getTestSize();

        mBgHandler.removeCallbacks(currentRunnable);

        currentRunnable = () -> {
            mNativeFedMLClientManager = new NativeFedMLClientManager();

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
                if (listener != null) {
                    listener.onTrainCompleted(trainModelPath, edgeId, clientIdx, trainSamples);
                }
            }
        };
        mBgHandler.post(currentRunnable);
    }

    public void resetTrain(final long runId, final int clientRunIndex) {
        runStateMap.put(runId + "_" + clientRunIndex, false);
        mBgHandler.removeCallbacks(currentRunnable);
        if (mNativeFedMLClientManager != null){
            mNativeFedMLClientManager = null;
        }
    }

    public void stopTrain() {
        runStateMap.clear();
        mBgHandler.removeCallbacks(currentRunnable);
        if (mNativeFedMLClientManager != null){
            mNativeFedMLClientManager = null;
        }
    }
}
