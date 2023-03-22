package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

/**
 * Train Start Listener
 */
public interface OnTrainStartListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        handleTrainStart(jsonMsg);
    }

    void handleTrainStart(JSONObject msgParams);
}
