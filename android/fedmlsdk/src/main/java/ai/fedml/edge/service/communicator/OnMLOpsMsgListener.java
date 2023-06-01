package ai.fedml.edge.service.communicator;

import org.json.JSONObject;

import ai.fedml.edge.utils.LogHelper;

public interface OnMLOpsMsgListener extends OnJsonReceivedListener {
    @Override
    default void onJsonReceived(String topic, JSONObject jsonMsg) {
        LogHelper.d("OnMLOpsMsgListener", "FedMLDebug. handleMLOpsMsg: " + jsonMsg.toString());
        handleMLOpsMsg(jsonMsg);
    }

    void handleMLOpsMsg(JSONObject msgParams);
}
