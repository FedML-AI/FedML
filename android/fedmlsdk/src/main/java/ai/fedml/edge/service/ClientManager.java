package ai.fedml.edge.service;

import com.amazonaws.mobileconnectors.s3.transferutility.TransferListener;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferState;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.service.communicator.OnTrainListener;
import ai.fedml.edge.service.communicator.message.BackModelMessage;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.service.communicator.message.TrainStatusMessage;
import ai.fedml.edge.service.component.ProfilerEventLogger;
import ai.fedml.edge.service.component.RemoteStorage;
import ai.fedml.edge.service.component.MetricsReporter;
import ai.fedml.edge.service.component.RuntimeLogger;
import ai.fedml.edge.service.entity.TrainingParams;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.StorageUtils;

import androidx.annotation.NonNull;

public final class ClientManager implements MessageDefine, OnTrainListener {
    private final EdgeCommunicator edgeCommunicator;
    private final TrainingExecutor mTrainer;
    private final MetricsReporter mReporter;
    private final RemoteStorage remoteStorage;
    private final ProfilerEventLogger eventLogger;
    private final RuntimeLogger mRuntimeLogger;
    private final long mEdgeId;
    private long mRunId;
    private final int mNumRounds;
    private int mClientIndex;
    private int mClientRound;
    private final String mDataset;
    private final int mBatchSize;
    private final double mLearningRate;
    private final int mEpochNum;
    private final int mTrainSize;
    private final int mTestSize;

    private boolean mHasSentOnlineMsg = false;

    private final Map<Long, Boolean> initStateMap;
    private final OnTrainProgressListener mOnTrainProgressListener;

    private interface OnUploadedListener {
        void onUploaded();
    }

    public ClientManager(final long edgeId, final long runId, final String strServerId, JSONObject hyperParameters,
                         @NonNull final OnTrainProgressListener onTrainProgressListener) {

        mEdgeId = edgeId;
        mRunId = runId;
        initStateMap = new ConcurrentHashMap<>();
        mOnTrainProgressListener = onTrainProgressListener;
        if (hyperParameters != null) {
            JSONObject trainArgs = hyperParameters.optJSONObject(TRAIN_ARGS);
            mNumRounds = trainArgs != null ? trainArgs.optInt(COMM_ROUND, 0) : 0;
            mBatchSize = trainArgs != null ? trainArgs.optInt(TRAIN_ARGS_BATCH_SIZE, 128) : 128;
            mLearningRate = trainArgs != null ? trainArgs.optDouble(TRAIN_ARGS_LR, 0.01f) : 0.01f;
            mEpochNum = trainArgs != null ? trainArgs.optInt(TRAIN_ARGS_EPOCH_NUM, 10) : 10;

            JSONObject dataArgs = hyperParameters.optJSONObject(DATA_ARGS);
            mDataset = dataArgs != null ? dataArgs.optString(DATASET_TYPE, "") : "";
            mTrainSize = dataArgs != null ? dataArgs.optInt(DATA_ARGS_TRAIN_SIZE, 600) : 600;
            mTestSize = dataArgs != null ? dataArgs.optInt(DATA_ARGS_TEST_SIZE, 100) : 100;
        } else {
            mNumRounds = 0;
            mDataset = "mnist";
            mBatchSize = 128;
            mLearningRate = 0.01f;
            mEpochNum = 10;
            mTrainSize = 600;
            mTestSize = 100;
        }
        LogHelper.d("ClientManager(%d, %d) dataSet=%s, hyperParameters=%s", edgeId, runId, mDataset, hyperParameters);
        mTrainer = new TrainingExecutor(onTrainProgressListener);
        eventLogger = new ProfilerEventLogger(edgeId, runId);
        edgeCommunicator = EdgeCommunicator.getInstance();
        mReporter = MetricsReporter.getInstance();
        remoteStorage = RemoteStorage.getInstance();
        mRuntimeLogger = new RuntimeLogger(edgeId, runId);
        mRuntimeLogger.initial();
        registerMessageReceiveHandlers(strServerId);
    }

    public void registerMessageReceiveHandlers(final String serverId) {
        final String runTopic = "fedml_" + mRunId + "_" + serverId + "_" + mEdgeId;
        edgeCommunicator.subscribe(runTopic, this);
    }

    private void send_init_online_msg(JSONObject params) {
        LogHelper.d("handle_message_check_status: %s", params.toString());
        // report MLOps that edge is OnLine now.
        mReporter.reportEdgeOnLine(mRunId, mEdgeId);
        // Notify MLOps with training status.
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_INITIALIZING);
    }

    @Override
    public void handle_message_finish(JSONObject params) {
        LogHelper.d("====================cleanup ====================");
        cleanup();
    }

    private void cleanup() {
        mReporter.reportEdgeFinished(mRunId, mEdgeId);
        finish();
        mRunId = 0;
    }

    @Override
    public void handleMessageConnectionReady(JSONObject params) {
        LogHelper.d("handleMessageConnectionReady: %d", mHasSentOnlineMsg?1:0);

        if (!mHasSentOnlineMsg) {
            mHasSentOnlineMsg = true;
            send_init_online_msg(params);
        }
    }

    @Override
    public void handleMessageInit(JSONObject params) {
        LogHelper.d("handleMessageInit: %s", params.toString());
        if (mEdgeId == 0) {
            return;
        }
        Boolean isInited = initStateMap.get(mRunId);
        if ( isInited != null && isInited ) {
            return;
        }
        String topic;
        String modelParams;
        try {
            topic = params.getString(TOPIC);
            modelParams = params.getString(MSG_ARG_KEY_MODEL_PARAMS);
            LogHelper.e("handleMessageInit modelParams (key) = %s", modelParams);
            mClientIndex = Integer.parseInt(params.getString(MSG_ARG_KEY_CLIENT_INDEX));
        } catch (JSONException e) {
            LogHelper.e(e, "handleTraining JSONException.");
            reportError();
            return;
        } catch (NumberFormatException e) {
            LogHelper.e(e, "handleTraining CLIENT_INDEX parseLong failed.");
            reportError();
            return;
        }
        initStateMap.put(mRunId, Boolean.TRUE);
        mClientRound = 0;
        handleTraining(topic, modelParams, 0);
        mClientRound += 1;
    }

    @Override
    public synchronized void handleMessageReceiveModelFromServer(JSONObject params) {
        LogHelper.d("handleMessageReceiveModelFromServer: %s", params.toString());
        if (mEdgeId == 0) {
            return;
        }
        LogHelper.d("[chaoyang] numRounds = %d, mClientRound = %d", mNumRounds, mClientRound);

        String topic;
        String modelParams;
        try {
            topic = params.getString(TOPIC);
            modelParams = params.getString(MSG_ARG_KEY_MODEL_PARAMS);
            mClientIndex = Integer.parseInt(params.getString(MSG_ARG_KEY_CLIENT_INDEX));
        } catch (JSONException e) {
            LogHelper.e(e, "handleTraining JSONException.");
            reportError();
            return;
        } catch (NumberFormatException e) {
            LogHelper.e(e, "handleTraining CLIENT_INDEX parseLong failed.");
            reportError();
            return;
        }

        if (mClientRound < mNumRounds) {
            handleTraining(topic, modelParams, mClientRound);
            mClientRound += 1;
        } else {
            downloadLastAggregatedModel(topic, modelParams, mClientRound);
            cleanup();
        }
    }

    @Override
    public void handle_message_check_status(JSONObject params) {
        send_init_online_msg(params);
    }

    private void handleTraining(final String topic, final String modelParams, final int clientRound) {
        eventLogger.logEventStarted("train", String.valueOf(clientRound));
        final String uuidKey = topic + "-" + UUID.randomUUID().toString().replace("-", "");
        final String trainModelPath = StorageUtils.getModelPath() + File.separator + uuidKey;
        LogHelper.d("modelParams（%s）", modelParams);
        LogHelper.d("trainModelPath（%s）", trainModelPath);
        remoteStorage.download(modelParams, new File(trainModelPath), new TransferListener() {
            private int reTryCnt = 3;

            @Override
            public void onStateChanged(int id, TransferState state) {
                LogHelper.d("download onStateChanged（%d, %s）", id, state);
                if (TransferState.COMPLETED == state) {
                    mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_TRAINING);
//                    mReporter.reportSystemMetric(mRunId, mEdgeId); // TODO: @zongchang.jie
                    final TrainingParams params = TrainingParams.builder()
                            .trainModelPath(trainModelPath).edgeId(mEdgeId).runId(mRunId)
                            .clientIdx(mClientIndex).dataSet(mDataset).clientRound(clientRound)
                            .batchSize(mBatchSize).learningRate(mLearningRate).trainSize(mTrainSize).testSize(mTestSize).epochNum(mEpochNum)
                            .listener((modelPath, edgeId, clientIdx, trainSamples) -> {
                                        eventLogger.logEventEnd("train", String.valueOf(clientRound));
                                        LogHelper.d("training is complete and start to sendModelToServer()");
                                        sendModelToServer(modelPath, edgeId, clientIdx, trainSamples, clientRound,
                                                () -> mOnTrainProgressListener.onProgressChanged(clientRound, 100.0f)
                                        );
                                    }
                            ).build();
                    mTrainer.training(params);
                } else if (TransferState.FAILED == state ) {
                    if ( reTryCnt > 0 ) {
                        remoteStorage.download(modelParams, new File(trainModelPath), this);
                        reTryCnt--;
                    } else {
                        reportError();
                    }
                }
            }

            @Override
            public void onProgressChanged(int id, long bytesCurrent, long bytesTotal) {
                LogHelper.d("download onProgressChanged(%d, %d, %d)", id, bytesCurrent, bytesTotal);
            }

            @Override
            public void onError(int id, Exception ex) {
                LogHelper.e(ex, "download onError(%d)", id);
                reportError();
            }
        });
    }

    private void downloadLastAggregatedModel(final String topic, final String modelParams, final int clientRound) {
        final String uuidKey = topic + "-" + UUID.randomUUID().toString().replace("-", "");
        final String trainModelPath = StorageUtils.getModelPath() + File.separator + uuidKey;
        LogHelper.d("modelParams（%s）", modelParams);
        LogHelper.d("trainModelPath（%s）", trainModelPath);
        remoteStorage.download(modelParams, new File(trainModelPath), new TransferListener() {
            private int reTryCnt = 3;

            @Override
            public void onStateChanged(int id, TransferState state) {
                LogHelper.d("download onStateChanged（%d, %s）", id, state);
                if (TransferState.COMPLETED == state) {
                    final TrainingParams params = TrainingParams.builder()
                            .trainModelPath(trainModelPath).edgeId(mEdgeId).runId(mRunId)
                            .clientIdx(mClientIndex).dataSet(mDataset).clientRound(clientRound)
                            .batchSize(mBatchSize).learningRate(mLearningRate).trainSize(mTrainSize).testSize(mTestSize).epochNum(mEpochNum)
                            .listener((modelPath, edgeId, clientIdx, trainSamples) -> {
                                    }
                            ).build();
                    // Todo: save the last aggregated model into the local training engine.
                    // mTrainer.training(params);
                } else if (TransferState.FAILED == state ) {
                    if ( reTryCnt > 0 ) {
                        remoteStorage.download(modelParams, new File(trainModelPath), this);
                        reTryCnt--;
                    }
                }
            }

            @Override
            public void onProgressChanged(int id, long bytesCurrent, long bytesTotal) {
                LogHelper.d("download onProgressChanged(%d, %d, %d)", id, bytesCurrent, bytesTotal);
            }

            @Override
            public void onError(int id, Exception ex) {
                LogHelper.e(ex, "download onError(%d)", id);
            }
        });
    }

    /**
     * upload Model to Server
     *
     * @param trainModelPath model path
     * @param edgeId         edge id
     * @param clientIdx      client index
     * @param trainSamples   train sample
     * @param clientRound    client Round
     * @param listener       OnUploadedListener
     */
    public void sendModelToServer(@NonNull final String trainModelPath, final long edgeId, final int clientIdx,
                                  final long trainSamples, final int clientRound, @NonNull final OnUploadedListener listener) {
        eventLogger.logEventStarted("comm_c2s", String.valueOf(clientRound));
        final String uuidS3Key = trainModelPath.substring(trainModelPath.lastIndexOf(File.separator) + 1);
        LogHelper.d("sendModelToServer uuidS3Key（%s）", uuidS3Key);
        remoteStorage.upload(uuidS3Key, new File(trainModelPath), new TransferListener() {
            private int reTryCnt = 3;

            @Override
            public void onStateChanged(int id, TransferState state) {
                LogHelper.d("upload onStateChanged（%d, %s）", id, state);
                if (state == TransferState.COMPLETED) {
                    sendModelMessage();
                    listener.onUploaded();
                } else if (TransferState.FAILED == state) {
                    if (reTryCnt > 0) {
                        remoteStorage.upload(uuidS3Key, new File(trainModelPath), this);
                        reTryCnt--;
                    }else {
                        reportError();
                    }
                }
            }

            @Override
            public void onProgressChanged(int id, long bytesCurrent, long bytesTotal) {
                LogHelper.d("upload onProgressChanged(%d, %d, %d)", id, bytesCurrent, bytesTotal);
            }

            @Override
            public void onError(int id, Exception ex) {
                LogHelper.e(ex, "upload onError(%d)", id);
                reportError();
            }

            private void sendModelMessage() {
                BackModelMessage modelEntity = BackModelMessage.builder().sender(edgeId).receiver(0)
                        .messageType(TrainStatusMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER)
                        .numSamples(trainSamples)
                        .modelParams(uuidS3Key)
                        .clientIdx(String.valueOf(clientIdx)).build();
                final String modelUploadTopic = "fedml_" + mRunId + "_" + edgeId;
                edgeCommunicator.sendMessage(modelUploadTopic, modelEntity);
                LogHelper.d("sendModelMessage is done.");
                mReporter.reportClientModelInfo(mRunId, edgeId, clientRound+1, uuidS3Key);
            }
        });
    }

    public void stopTrain() {
        LogHelper.i("stop train");
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_KILLED);
        mTrainer.stopTrain();
    }

    public void stopTrainWithoutReportStatus() {
        LogHelper.i("stop train without status reporting.");
        mTrainer.stopTrain();
    }

    private void finish() {
        LogHelper.i("Training finished for master client");
        mReporter.reportClientStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FINISHED);

        // Notify MLOps with the finished message
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FINISHED);
    }

    private void reportError() {
        LogHelper.i("Report training error!");
        stopTrain();
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FAILED);
    }

}
