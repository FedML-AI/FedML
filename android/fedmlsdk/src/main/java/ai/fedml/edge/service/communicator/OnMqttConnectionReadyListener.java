package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

import ai.fedml.edge.utils.LogHelper;

public interface OnMqttConnectionReadyListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        LogHelper.d("OnMqttConnectionReadyListener", "FedMLDebug. handleConnectionReady: " + jsonMsg.toString());
        handleConnectionReady(jsonMsg);
    }

    void handleConnectionReady(JSONObject msgParams);
}
