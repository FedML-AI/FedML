package ai.fedml.edge.service.component;

import org.json.JSONException;
import org.json.JSONObject;

import ai.fedml.edge.constants.FedMqttTopic;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.TimeUtils;

public class ProfilerEventLogger {

    private static final int EVENT_TYPE_STARTED = 0;
    private static final int EVENT_TYPE_ENDED = 1;

    private final EdgeCommunicator edgeCommunicator;
    private final long mEdgeId;
    private final long mRunId;

    public ProfilerEventLogger(final long edgeId, final long runId) {
        mEdgeId = edgeId;
        mRunId = runId;
        edgeCommunicator = EdgeCommunicator.getInstance();
    }

    public void logEventStarted(final String eventName, final String eventValue) {
        logEventStarted(eventName, eventValue, null);
    }

    public void logEventEnd(final String eventName, final String eventValue) {
        logEventEnd(eventName, eventValue, null);
    }

    private void logEventStarted(final String eventName, final String value, final Long eventEdgeId) {
        final long edgeId = eventEdgeId == null ? mEdgeId : eventEdgeId;
        JSONObject jsonObject = buildEventMessage(mRunId, edgeId, EVENT_TYPE_STARTED, eventName, value);
        edgeCommunicator.sendMessage(FedMqttTopic.EVENT, jsonObject.toString());
    }

    private void logEventEnd(final String eventName, final String value, final Long eventEdgeId) {
        final long edgeId = eventEdgeId == null ? mEdgeId : eventEdgeId;
        JSONObject jsonObject = buildEventMessage(mRunId, edgeId, EVENT_TYPE_ENDED, eventName, value);
        edgeCommunicator.sendMessage(FedMqttTopic.EVENT, jsonObject.toString());
    }

    private static JSONObject buildEventMessage(final long runId, final long edgeId, final int event_type,
                                                final String name, final String value) {
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("run_id", runId);
            jsonObject.put("edge_id", edgeId);
            jsonObject.put("event_name", name);
            jsonObject.put("event_value", value);
            String timeKey = "started_time";
            if (EVENT_TYPE_STARTED == event_type) {
                timeKey = "started_time";
            } else if (EVENT_TYPE_ENDED == event_type) {
                timeKey = "ended_time";
            }
            jsonObject.put(timeKey, TimeUtils.getAccurateTime());
        } catch (JSONException e) {
            LogHelper.e(e, "buildEventMessage(%d, %s, %s)", event_type, name, value);
        }
        return jsonObject;
    }
}
