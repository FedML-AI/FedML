package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

import ai.fedml.edge.utils.LogHelper;

/**
 * Train Start Listener
 */
public interface OnTrainStartListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        LogHelper.d("OnTrainStartListener", "FedMLDebug. handleTrainStart: " + jsonMsg);
        handleTrainStart(jsonMsg);
    }

    void handleTrainStart(JSONObject msgParams);
}
