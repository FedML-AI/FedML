package ai.fedml.edge.service;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;

import org.json.JSONObject;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.constants.FedMqttTopic;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.service.communicator.OnMLOpsMsgListener;
import ai.fedml.edge.service.communicator.OnMqttConnectionReadyListener;
import ai.fedml.edge.service.communicator.OnTrainStartListener;
import ai.fedml.edge.service.communicator.OnTrainStopListener;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.service.component.DeviceInfoReporter;
import ai.fedml.edge.service.component.MetricsReporter;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;

import androidx.annotation.NonNull;

public final class ClientAgentManager implements MessageDefine {
    private final OnTrainProgressListener onTrainProgressListener;
    private final OnTrainingStatusListener onTrainingStatusListener;
    private final EdgeCommunicator edgeCommunicator;
    private final MetricsReporter mReporter;
    private long mEdgeId = 0;

    private final DeviceInfoReporter mDeviceInfoReporter;

    private volatile long mRunId = 0;
    private final Gson mGson;

    private ClientManager mClientManager;

    public ClientAgentManager(String edgeID, @NonNull final OnTrainingStatusListener onTrainingStatusListener,
                              @NonNull final OnTrainProgressListener onTrainProgressListener) {
        mEdgeId = Long.parseLong(edgeID);
        this.onTrainingStatusListener = onTrainingStatusListener;
        this.onTrainProgressListener = onTrainProgressListener;
        edgeCommunicator = EdgeCommunicator.getInstance();

        mReporter = MetricsReporter.getInstance();
        mReporter.setEdgeCommunicator(edgeCommunicator);
        mReporter.setTrainingStatusListener(onTrainingStatusListener);

        mGson = new GsonBuilder().setPrettyPrinting().create();
        SharePreferencesData.clearHyperParameters();

        edgeCommunicator.addListener((OnMqttConnectionReadyListener) this::handleMqttConnectionReady);

        mDeviceInfoReporter = new DeviceInfoReporter(mEdgeId, edgeCommunicator);
        mDeviceInfoReporter.start();
    }

    public void start() {
        edgeCommunicator.connect();
    }

    /**
     * register handlers
     *
     * @param edgeId edge id
     */
    public void registerMessageReceiveHandlers(final long edgeId) {
        LogHelper.i("FedMLDebug. registerMessageReceiveHandlers. mReporter = " + mReporter + ", edgeId = " + edgeId);

        edgeCommunicator.subscribe(FedMqttTopic.startTrain(edgeId), (OnTrainStartListener) this::handleTrainStart);
        edgeCommunicator.subscribe(FedMqttTopic.stopTrain(edgeId), (OnTrainStopListener) this::handleTrainStop);
        edgeCommunicator.subscribe(FedMqttTopic.REPORT_DEVICE_STATUS, (OnMLOpsMsgListener) this::handleMLOpsMsg);
        edgeCommunicator.subscribe(FedMqttTopic.exitTrainWithException(edgeId), (OnMLOpsMsgListener) this::handleTrainException);
    }

    private void handleMqttConnectionReady(JSONObject msgParams) {
        LogHelper.i("FedMLDebug. handleMqttConnectionReady");
        mReporter.syncClientStatus(mEdgeId);
        registerMessageReceiveHandlers(mEdgeId);
    }

    private void handleTrainStart(JSONObject msgParams) {
        LogHelper.i("onStartTrain: %s", msgParams.toString());
        if (mEdgeId == 0) {
            LogHelper.w("handleTrainStart but mEdgeId is 0");
            return;
        }

        // TODO: waiting dataset split, then download the dataset package and Training Client App

        long runId = msgParams.optLong("runId", 0);
        mRunId = runId;

        JSONObject hyperParameters = null;
        final String strServerId = msgParams.optString(TRAIN_SERVER_ID);
        JSONObject runConfigJson = msgParams.optJSONObject(RUN_CONFIG);
        if (runConfigJson != null) {
            hyperParameters = runConfigJson.optJSONObject(HYPER_PARAMETERS_CONFIG);
            if (hyperParameters != null) {
                JsonElement jsonElement = JsonParser.parseString(hyperParameters.toString());
                SharePreferencesData.saveHyperParameters(mGson.toJson(jsonElement));
            } else {
                SharePreferencesData.clearHyperParameters();
            }
            if(this.onTrainingStatusListener != null) {
                this.onTrainingStatusListener.onStatusChanged(KEY_CLIENT_STATUS_INITIALIZING);
            }
        }
        // Launch Training Client
        mClientManager = new ClientManager(mEdgeId, runId, strServerId, hyperParameters, onTrainProgressListener);
    }

    private void handleTrainStop(JSONObject msgParams) {
        LogHelper.i("handleTrainStop :%s", msgParams.toString());
        if(this.onTrainingStatusListener != null) {
            this.onTrainingStatusListener.onStatusChanged(KEY_CLIENT_STATUS_KILLED);
        }
        // Stop Training Client
        if (mClientManager != null) {
            mClientManager.stopTrain();
            mClientManager = null;
            LogHelper.i("FedMLDebug mClientManager is killed");
        }
        mReporter.reportTrainingStatus(0, mEdgeId, KEY_CLIENT_STATUS_IDLE);
        mRunId = 0;
    }

    private void handleMLOpsMsg(JSONObject msgParams) {
        LogHelper.i("handleMLOpsMsg :%s", msgParams.toString());
        mReporter.reportClientActiveStatus(mEdgeId);
    }

    private void handleTrainException(JSONObject msgParams) {
        LogHelper.i("handleTrainException :%s", msgParams.toString());
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FAILED);

        if (mClientManager != null) {
            mClientManager.stopTrainWithoutReportStatus();
            mClientManager = null;
        }
        mRunId = 0;
    }

}
