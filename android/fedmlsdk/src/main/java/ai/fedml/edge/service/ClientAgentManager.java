package ai.fedml.edge.service;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;

import org.json.JSONObject;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
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
        LogHelper.d("FedMLDebug. registerMessageReceiveHandlers. mReporter = " + mReporter + ", edgeId = " + edgeId);

        final String startTrainTopic = "flserver_agent/" + edgeId + "/start_train";
        edgeCommunicator.subscribe(startTrainTopic, (OnTrainStartListener) this::handleTrainStart);

        final String stopTrainTopic = "flserver_agent/" + edgeId + "/stop_train";
        edgeCommunicator.subscribe(stopTrainTopic, (OnTrainStopListener) this::handleTrainStop);

        final String MLOpsQueryStatusTopic = "mlops/report_device_status";
        edgeCommunicator.subscribe(MLOpsQueryStatusTopic, (OnMLOpsMsgListener) this::handleMLOpsMsg);

        final String exitTrainWithExceptionTopic = "flserver_agent/" + edgeId + "/exit_train_with_exception";
        edgeCommunicator.subscribe(exitTrainWithExceptionTopic, (OnMLOpsMsgListener) this::handleTrainException);
    }

    private void handleMqttConnectionReady(JSONObject msgParams) {
        LogHelper.d("FedMLDebug. handleMqttConnectionReady");
        mReporter.syncClientStatus(mEdgeId);
        registerMessageReceiveHandlers(mEdgeId);
    }

    private void handleTrainStart(JSONObject msgParams) {
        LogHelper.d("onStartTrain: %s", msgParams.toString());
        if (mEdgeId == 0) {
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
        LogHelper.d("handleTrainStop :%s", msgParams.toString());
        if(this.onTrainingStatusListener != null) {
            this.onTrainingStatusListener.onStatusChanged(KEY_CLIENT_STATUS_KILLED);
        }
        // Stop Training Client
        if (mClientManager != null) {
            mClientManager.stopTrain();
            mClientManager = null;
            LogHelper.d("FedMLDebug mClientManager is killed");
        }
        mReporter.reportTrainingStatus(0, mEdgeId, KEY_CLIENT_STATUS_IDLE);
        mRunId = 0;
    }

    private void handleMLOpsMsg(JSONObject msgParams) {
        LogHelper.d("handleMLOpsMsg :%s", msgParams.toString());
        mReporter.reportClientActiveStatus(mEdgeId);
    }

    private void handleTrainException(JSONObject msgParams) {
        LogHelper.d("handleTrainException :%s", msgParams.toString());
        mReporter.reportTrainingStatus(mRunId, mEdgeId, KEY_CLIENT_STATUS_FAILED);

        if (mClientManager != null) {
            mClientManager.stopTrainWithoutReportStatus();
            mClientManager = null;
        }
        mRunId = 0;
    }
}
