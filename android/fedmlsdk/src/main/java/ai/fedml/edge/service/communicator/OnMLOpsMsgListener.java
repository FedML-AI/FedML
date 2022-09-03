package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

public interface OnMLOpsMsgListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        handleMLOpsMsg(jsonMsg);
    }

    void handleMLOpsMsg(JSONObject msgParams);
}
