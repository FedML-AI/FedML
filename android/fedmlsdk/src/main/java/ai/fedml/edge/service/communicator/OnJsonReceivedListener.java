package ai.fedml.edge.service.communicator;

import org.json.JSONException;
import org.json.JSONObject;

import ai.fedml.edge.utils.LogHelper;

public interface OnJsonReceivedListener extends OnReceivedListener {
    @Override
    default void onReceived(String topic, byte[] payload) {
        try {
            JSONObject jsonMsg = new JSONObject(new String(payload));
            onJsonReceived(topic, jsonMsg);
        } catch (JSONException e) {
            LogHelper.e(e, "Received msg cannot transfer to JSON.");
        }
    }

    void onJsonReceived(String topic, JSONObject jsonMsg);
}
