package ai.fedml.edge.service;

import com.amazonaws.mobileconnectors.s3.transferutility.TransferListener;
import com.amazonaws.mobileconnectors.s3.transferutility.TransferState;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.constants.FedMqttTopic;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.service.communicator.OnTrainErrorListener;
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

public final class ClientManager implements MessageDefine, OnTrainListener, OnTrainErrorListener {
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

    private boolean mIsTrainingStopped = false;

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
        LogHelper.i("ClientManager(%d, %d) dataSet=%s, hyperParameters=%s", edgeId, runId, mDataset, hyperParameters);
        mTrainer = new TrainingExecutor(onTrainProgressListener);
        eventLogger = new ProfilerEventLogger(edgeId, runId);
        edgeCommunicator = EdgeCommunicator.getInstance();
        mReporter = MetricsReporter.getInstance();
        remoteStorage = RemoteStorage.getInstance();
        mRuntimeLogger = new RuntimeLogger(edgeId, runId);
        mRuntimeLogger.start();
        registerMessageReceiveHandlers(strServerId);
    }

    public void registerMessageReceiveHandlers(final String serverId) {
        edgeCommunicator.subscribe(FedMqttTopic.run(mRunId, serverId, mEdgeId), this);
    }

    private void sendInitOnlineMsg(JSONObject params) {
        LogHelper.i("handle_message_check_status: %s", params.toString());
        // report MLOps that edge is OnLine now.
        mReporter.reportEdgeOnLine(mRunId, mEdgeId);
        // Notify MLOps with training status.
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_INITIALIZING);
    }

    @Override
    public void handleMessageFinish(JSONObject params) {
        LogHelper.i("====================cleanup ====================");
        finishRun();
    }

    @Override
    public void handleMessageConnectionReady(JSONObject params) {
        LogHelper.i("handleMessageConnectionReady: %d", mHasSentOnlineMsg ? 1 : 0);

        if (!mHasSentOnlineMsg) {
            mHasSentOnlineMsg = true;
            sendInitOnlineMsg(params);
        }
    }

    @Override
    public void handleMessageInit(JSONObject params) {
        LogHelper.i("handleMessageInit: %s", params.toString());
        if (mEdgeId == 0) {
            return;
        }
        Boolean isInited = initStateMap.get(mRunId);
        if (isInited != null && isInited) {
            return;
        }
        String topic;
        String modelParams;
        try {
            topic = params.getString(TOPIC);
            modelParams = params.getString(MSG_ARG_KEY_MODEL_PARAMS);
            LogHelper.i("FedMLDebug. handleMessageInit modelParams (key) = %s", modelParams);
            mClientIndex = Integer.parseInt(params.getString(MSG_ARG_KEY_CLIENT_INDEX));
        } catch (JSONException e) {
            LogHelper.e(e, "handleMessageInit failed");
            reportError();
            return;
        } catch (NumberFormatException e) {
            LogHelper.e(e, "handleTraining CLIENT_INDEX parseLong failed");
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
        LogHelper.i("handleMessageReceiveModelFromServer: %s", params.toString());
        if (mIsTrainingStopped) {
            LogHelper.i("FedMLDebug. handleMessageReceiveModelFromServer() training run (%s) is already stopped", mRunId);
            return;
        }
        if (mEdgeId == 0) {
            return;
        }

        String topic;
        String modelParams;
        try {
            topic = params.getString(TOPIC);
            modelParams = params.getString(MSG_ARG_KEY_MODEL_PARAMS);
            mClientIndex = Integer.parseInt(params.getString(MSG_ARG_KEY_CLIENT_INDEX));
        } catch (JSONException e) {
            LogHelper.e(e, "handleMessageReceiveModelFromServer failed.");
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
            finishRun();
        }
    }

    @Override
    public void handleMessageCheckStatus(JSONObject params) {
        sendInitOnlineMsg(params);
    }

    private void handleTraining(final String topic, final String modelParams, final int clientRound) {
        LogHelper.i("FedMLDebug. handleTraining topic（%s）, modelParams（%s）,clientRound(%d)", topic, modelParams, clientRound);
        if (mIsTrainingStopped) {
            LogHelper.i("FedMLDebug. handleTraining() training run (%s) is already stopped", mRunId);
            return;
        }
        eventLogger.logEventStarted("train", String.valueOf(clientRound));
        final String uuidKey = topic + "-" + UUID.randomUUID().toString().replace("-", "");
        final String trainModelPath = StorageUtils.getModelPath() + File.separator + uuidKey;

        remoteStorage.download(modelParams, new File(trainModelPath), new TransferListener() {
            private int reTryCnt = 3;

            @Override
            public void onStateChanged(int id, TransferState state) {
                LogHelper.d("download onStateChanged（%d, %s）", id, state);
                if (TransferState.COMPLETED == state) {
                    mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_TRAINING);
                    final TrainingParams params = TrainingParams.builder()
                            .trainModelPath(trainModelPath).edgeId(mEdgeId).runId(mRunId)
                            .clientIdx(mClientIndex).dataSet(mDataset).clientRound(clientRound)
                            .batchSize(mBatchSize).learningRate(mLearningRate).trainSize(mTrainSize).testSize(mTestSize).epochNum(mEpochNum)
                            .errorListener(ClientManager.this)
                            .listener((modelPath, edgeId, clientIdx, trainSamples) -> {
                                        eventLogger.logEventEnd("train", String.valueOf(clientRound));
                                        LogHelper.i("FedMLDebug. training is complete and start to sendModelToServer() modelPath = " + modelPath);
                                        sendModelToServer(modelPath, edgeId, clientIdx, trainSamples, clientRound,
                                                () -> mOnTrainProgressListener.onProgressChanged(clientRound, 100.0f)
                                        );
                                    }
                            ).build();
                    try {
                        mTrainer.training(params);
                    } catch (IOException e) {
                        LogHelper.e(e, "FedMLDebug. training failed.");
                        reportError();
                    }
                } else if (TransferState.FAILED == state) {
                    if (reTryCnt > 0) {
                        remoteStorage.download(modelParams, new File(trainModelPath), this);
                        reTryCnt--;
                    } else {
                        LogHelper.e("download transfer state is failed");
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
                if (reTryCnt > 0) {
                    LogHelper.w(ex, "download onError(%d)", id);
                    remoteStorage.download(modelParams, new File(trainModelPath), this);
                    reTryCnt--;
                } else {
                    LogHelper.e(ex, "download onError(%d)", id);
                    reportError();
                }
            }
        });
    }

    private void downloadLastAggregatedModel(final String topic, final String modelParams, final int clientRound) {
        LogHelper.i("FedMLDebug. downloadLastAggregatedModel topic（%s, modelParams(%s), clientRound(%d)", topic, modelParams, clientRound);
        final String uuidKey = topic + "-" + UUID.randomUUID().toString().replace("-", "");
        final String trainModelPath = StorageUtils.getModelPath() + File.separator + uuidKey;
        remoteStorage.download(modelParams, new File(trainModelPath), new TransferListener() {
            private int reTryCnt = 3;

            @Override
            public void onStateChanged(int id, TransferState state) {
//                LogHelper.d("download onStateChanged（%d, %s）", id, state);
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
                } else if (TransferState.FAILED == state) {
                    if (reTryCnt > 0) {
                        remoteStorage.download(modelParams, new File(trainModelPath), this);
                        reTryCnt--;
                    }
                }
            }

            @Override
            public void onProgressChanged(int id, long bytesCurrent, long bytesTotal) {
//                LogHelper.d("download onProgressChanged(%d, %d, %d)", id, bytesCurrent, bytesTotal);
            }

            @Override
            public void onError(int id, Exception ex) {
                if (reTryCnt > 0) {
                    LogHelper.w(ex, "download onError(%d)", id);
                    remoteStorage.download(modelParams, new File(trainModelPath), this);
                    reTryCnt--;
                } else {
                    LogHelper.e(ex, "download onError(%d)", id);
                    reportError();
                }
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
        if (mIsTrainingStopped) {
            LogHelper.i("FedMLDebug. sendModelToServer() training run (%s) is already stopped", mRunId);
            return;
        }
        eventLogger.logEventStarted("comm_c2s", String.valueOf(clientRound));
        final String uuidS3Key = trainModelPath.substring(trainModelPath.lastIndexOf(File.separator) + 1);
        LogHelper.d("FedMLDebug. sendModelToServer uuidS3Key（%s), trainModelPath (%s)", uuidS3Key, trainModelPath);
        remoteStorage.upload(uuidS3Key, new File(trainModelPath), new TransferListener() {
            private int reTryCnt = 3;

            @Override
            public void onStateChanged(int id, TransferState state) {
                if (state == TransferState.COMPLETED) {
                    sendModelMessage();
                    listener.onUploaded();
                } else if (TransferState.FAILED == state) {
                    if (reTryCnt > 0) {
                        remoteStorage.upload(uuidS3Key, new File(trainModelPath), this);
                        reTryCnt--;
                    } else {
                        LogHelper.e("send model to server failed");
                        reportError();
                    }
                }
            }

            @Override
            public void onProgressChanged(int id, long bytesCurrent, long bytesTotal) {
//                LogHelper.d("upload onProgressChanged(%d, %d, %d)", id, bytesCurrent, bytesTotal);
            }

            @Override
            public void onError(int id, Exception ex) {
                if (reTryCnt > 0) {
                    LogHelper.w(ex, "upload onError(%d)", id);
                    remoteStorage.upload(uuidS3Key, new File(trainModelPath), this);
                    reTryCnt--;
                } else {
                    LogHelper.e(ex, "upload onError(%d)", id);
                    reportError();
                }
            }

            private void sendModelMessage() {
                BackModelMessage modelEntity = BackModelMessage.builder().sender(edgeId).receiver(0)
                        .messageType(TrainStatusMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER)
                        .numSamples(trainSamples)
                        .modelParams(uuidS3Key)
                        .clientIdx(String.valueOf(clientIdx)).build();
                edgeCommunicator.sendMessage(FedMqttTopic.modelUpload(mRunId, mEdgeId), modelEntity);
                mReporter.reportClientModelInfo(mRunId, edgeId, clientRound + 1, uuidS3Key);
            }
        });
    }

    public void stopTrain() {
        LogHelper.i("FedMLDebug. stop train");
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_KILLED);

        mTrainer.stopTrain();
        cleanUpRun();
        mIsTrainingStopped = true;
        mRunId = 0;
    }

    public void stopTrainWithoutReportStatus() {
        LogHelper.i("FedMLDebug. stop train without status reporting.");
        mTrainer.stopTrain();
        cleanUpRun();
        mIsTrainingStopped = true;
        mRunId = 0;
    }

    @Override
    public void onTrainError(Throwable throwable) {
        LogHelper.e(throwable, "onTrainError");
        reportError();
    }

    private void finishRun() {
        mReporter.reportEdgeFinished(mRunId, mEdgeId);

        LogHelper.i("Training finished for master client");
        mReporter.reportClientStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FINISHED);

        // Notify MLOps with the finished message
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FINISHED);
        mRunId = 0;
        cleanUpRun();
    }

    private void reportError() {
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FAILED);
        stopTrain();
    }

    private void cleanUpRun() {
        mRuntimeLogger.release();
    }

}
