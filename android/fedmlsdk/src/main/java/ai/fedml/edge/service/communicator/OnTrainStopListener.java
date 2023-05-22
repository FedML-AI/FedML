package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

import ai.fedml.edge.utils.LogHelper;

public interface OnTrainStopListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        LogHelper.d("OnTrainStopListener", "FedMLDebug. handleTrainStop: " + jsonMsg);
        handleTrainStop(jsonMsg);
    }

    void handleTrainStop(JSONObject msgParams);
}
