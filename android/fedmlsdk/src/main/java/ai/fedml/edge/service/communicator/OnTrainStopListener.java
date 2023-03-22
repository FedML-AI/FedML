package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

public interface OnTrainStopListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        handleTrainStop(jsonMsg);
    }

    void handleTrainStop(JSONObject msgParams);
}
