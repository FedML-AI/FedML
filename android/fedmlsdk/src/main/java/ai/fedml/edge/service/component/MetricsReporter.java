package ai.fedml.edge.service.component;

import org.json.JSONException;
import org.json.JSONObject;

import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.service.communicator.message.TrainStatusMessage;
import ai.fedml.edge.utils.LogHelper;
import androidx.annotation.NonNull;
import androidx.annotation.StringRes;

public class MetricsReporter implements MessageDefine {
    private final EdgeCommunicator edgeCommunicator;
    private int mClientStatus = KEY_CLIENT_STATUS_IDLE;
    private OnTrainingStatusListener mOnTrainingStatusListener;

    private final static class LazyHolder {
        private static final MetricsReporter sInstance = new MetricsReporter();
    }

    public static MetricsReporter getInstance() {
        return MetricsReporter.LazyHolder.sInstance;
    }

    public MetricsReporter() {
        edgeCommunicator = EdgeCommunicator.getInstance();
    }

    public void setTrainingStatusListener(@NonNull final OnTrainingStatusListener onTrainingStatusListener) {
        mOnTrainingStatusListener = onTrainingStatusListener;
    }

    public int getClientStatus() {
        return mClientStatus;
    }

    public boolean reportEdgeOnLine(final long runId, final long edgeId) {
        TrainStatusMessage trainStatus = TrainStatusMessage.builder().sender(edgeId).receiver(0)
                .messageType(TrainStatusMessage.MSG_TYPE_C2S_CLIENT_STATUS)
                .status("ONLINE").os(MSG_CLIENT_OS_ANDROID).build();
        final String onLineTopic = "fedml_" + runId + "_" + edgeId;
        return edgeCommunicator.sendMessage(onLineTopic, trainStatus);
    }

    public boolean reportEdgeFinished(final long runId, final long edgeId) {
        TrainStatusMessage trainStatus = TrainStatusMessage.builder().sender(edgeId).receiver(0)
                .messageType(TrainStatusMessage.MSG_TYPE_C2S_CLIENT_STATUS)
                .status("FINISHED").os(MSG_CLIENT_OS_ANDROID).build();
        final String onLineTopic = "fedml_" + runId + "_" + edgeId;
        return edgeCommunicator.sendMessage(onLineTopic, trainStatus);
    }

    public void reportClientStatus(final long runId, final long edgeId, final int status) {
        notifyClientStatus(status);
        final String topic = "fl_client/mlops/" + edgeId + "/status";
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(REPORT_STATUS, CLIENT_STATUS_MAP.get(status));
        } catch (JSONException e) {
            LogHelper.e(e, "reportClientStatus(%d, %d, %d)", runId, edgeId, status);
        }
        edgeCommunicator.sendMessage(topic, jsonObject.toString());
    }

    public void reportTrainingStatus(final long runId, final long edgeId, final int status) {
        notifyClientStatus(status);
        final String topicName4WebUI = "fl_client/mlops/status";
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(REPORT_STATUS, CLIENT_STATUS_MAP.get(status));
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %d)", edgeId, status);
        }
        edgeCommunicator.sendMessage(topicName4WebUI, jsonObject.toString());

        final String topicName4Run = "fl_run/fl_client/mlops/status";
        edgeCommunicator.sendMessage(topicName4Run, jsonObject.toString());

        if (status == KEY_CLIENT_STATUS_FAILED) {
            reportClientException(runId, edgeId, status);
        }
    }

    public void reportClientActiveStatus(final long edgeId) {
        JSONObject jsonObject = new JSONObject();
        if (mClientStatus != KEY_CLIENT_STATUS_OFFLINE &&
                mClientStatus != KEY_CLIENT_STATUS_IDLE &&
                mClientStatus != KEY_CLIENT_STATUS_FINISHED) {
            return;
        }
        try {
            notifyClientStatus(KEY_CLIENT_STATUS_IDLE);
            jsonObject.put("ID", edgeId);
            jsonObject.put(REPORT_STATUS, MSG_MLOPS_CLIENT_STATUS_IDLE);
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %s)", edgeId, MSG_MLOPS_CLIENT_STATUS_IDLE);
        }
        edgeCommunicator.sendMessage(MQTT_REPORT_ACTIVE_STATUS_TOPIC, jsonObject.toString());
    }

    public void reportClientModelInfo(final long runId, final long edgeId, final int clientRound, final String model) {
        final String topicName = "fl_server/mlops/client_model";
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(ROUND_IDX, clientRound);
            jsonObject.put(CLIENT_MODEL_ADDRESS, model);
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %s, %s)", edgeId, clientRound, model);
        }
        edgeCommunicator.sendMessage(topicName, jsonObject.toString());
    }

    public void reportTrainingMetric(long edgeId, long runId, float accuracy, float loss) {
        final String topicMetrics = "fl_client/mlops/training_progress_and_eval";
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("run_id", runId);
            jsonObject.put("edge_id", edgeId);
            jsonObject.put("accuracy", accuracy);
            jsonObject.put("loss", loss);
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingMetric(%s)", edgeId);
        }
        edgeCommunicator.sendMessage(topicMetrics, jsonObject.toString());
    }

    public void reportSystemMetric(final long runId, final long edgeId) {
        final String topicSysMetrics = "fl_client/mlops/system_performance";
        JSONObject jsonObject = new JSONObject();
        final SysStats sysStats = SysStats.getInstance();
        try {
            jsonObject.put("run_id", runId);
            jsonObject.put("edge_id", edgeId);
            jsonObject.put("cpu_utilization", sysStats.getCpuUtilization());
            jsonObject.put("process_cpu_threads_in_use", sysStats.getProcessCpuThreadsInUse());
            SysStats.MemoryStats memoryStats = sysStats.getMemoryInfo();
            if (memoryStats != null) {
                jsonObject.put("SystemMemoryUtilization", memoryStats.getMemoryUtilization());
                jsonObject.put("process_memory_in_use", memoryStats.getMemoryInUse());
                jsonObject.put("process_memory_in_use_size", memoryStats.getMemoryInUseSize());
                jsonObject.put("process_memory_available", memoryStats.getMemoryAvailable());
            }
            jsonObject.put("disk_utilization", sysStats.getDiskUtilization());
            jsonObject.put("network_traffic", sysStats.getNetworkTraffic());
        } catch (JSONException e) {
            LogHelper.e(e, "reportSystemMetric(%s)", edgeId);
        }
        edgeCommunicator.sendMessage(topicSysMetrics, jsonObject.toString());
    }

    private void notifyClientStatus(final int status) {
        LogHelper.d("notifyClientStatus [%d]", status);
        mClientStatus = status;
        if (mOnTrainingStatusListener != null) {
            mOnTrainingStatusListener.onStatusChanged(status);
        }
    }

    public void reportClientException(final long runId, final long edgeId, int status) {
        notifyClientStatus(status);
        final String topicException = "flserver_agent/" + runId + "/client_exit_train_with_exception";
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(REPORT_STATUS, CLIENT_STATUS_MAP.get(status));
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %d)", edgeId, status);
        }
        edgeCommunicator.sendMessage(topicException, jsonObject.toString());
    }
}
