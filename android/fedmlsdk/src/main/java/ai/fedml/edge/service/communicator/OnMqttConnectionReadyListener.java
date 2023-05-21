package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

public interface OnMqttConnectionReadyListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        handleConnectionReady(jsonMsg);
    }

    void handleConnectionReady(JSONObject msgParams);
}
