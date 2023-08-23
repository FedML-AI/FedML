package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

import ai.fedml.edge.utils.LogHelper;

/**
 * Train Start Listener
 */
public interface OnTrainStartListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        LogHelper.i("FedMLDebug. OnTrainStartListener handleTrainStart:%s", jsonMsg.toString());
        handleTrainStart(jsonMsg);
    }

    void handleTrainStart(JSONObject msgParams);
}
