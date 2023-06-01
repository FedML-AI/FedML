package ai.fedml.edge.service.component;

import org.json.JSONException;
import org.json.JSONObject;

import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.constants.FedMqttTopic;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.service.communicator.message.TrainStatusMessage;
import ai.fedml.edge.utils.LogHelper;
import androidx.annotation.NonNull;

public class MetricsReporter implements MessageDefine, MessageDefine.ClientStatus {
    private EdgeCommunicator edgeCommunicator;
    private int mClientStatus = KEY_CLIENT_STATUS_IDLE;
    private long mRunId = 0;
    private OnTrainingStatusListener mOnTrainingStatusListener;

    private final static class LazyHolder {
        private static final MetricsReporter sInstance = new MetricsReporter();
    }

    public static MetricsReporter getInstance() {
        return MetricsReporter.LazyHolder.sInstance;
    }

    public MetricsReporter() {
        edgeCommunicator = null;
    }

    public void setEdgeCommunicator(EdgeCommunicator communicator) {
        edgeCommunicator = communicator;
    }

    public void setTrainingStatusListener(@NonNull final OnTrainingStatusListener onTrainingStatusListener) {
        mOnTrainingStatusListener = onTrainingStatusListener;
    }

    public boolean reportEdgeOnLine(final long runId, final long edgeId) {
        TrainStatusMessage trainStatus = TrainStatusMessage.builder().sender(edgeId).receiver(0)
                .messageType(TrainStatusMessage.MSG_TYPE_C2S_CLIENT_STATUS)
                .status("ONLINE").os(MSG_CLIENT_OS_ANDROID).build();
        return edgeCommunicator.sendMessage(FedMqttTopic.online(runId, edgeId), trainStatus);
    }

    public boolean reportEdgeFinished(final long runId, final long edgeId) {
        TrainStatusMessage trainStatus = TrainStatusMessage.builder().sender(edgeId).receiver(0)
                .messageType(TrainStatusMessage.MSG_TYPE_C2S_CLIENT_STATUS)
                .status("FINISHED").os(MSG_CLIENT_OS_ANDROID).build();
        return edgeCommunicator.sendMessage(FedMqttTopic.online(runId, edgeId), trainStatus);
    }

    public void reportClientStatus(final long runId, final long edgeId, final int status) {
        notifyClientStatus(runId, status);
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(REPORT_STATUS, CLIENT_STATUS_MAP.get(status));
        } catch (JSONException e) {
            LogHelper.e(e, "reportClientStatus(%d, %d, %d)", runId, edgeId, status);
        }
        edgeCommunicator.sendMessage(FedMqttTopic.flclientStatus(edgeId), jsonObject.toString());
    }

    public void reportTrainingStatus(final long runId, final long edgeId, final int status) {
        notifyClientStatus(runId, status);
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(REPORT_STATUS, CLIENT_STATUS_MAP.get(status));
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %d)", edgeId, status);
        }
        edgeCommunicator.sendMessage(FedMqttTopic.STATUS, jsonObject.toString());

        edgeCommunicator.sendMessage(FedMqttTopic.RUN_STATUS, jsonObject.toString());

        if (status == KEY_CLIENT_STATUS_FAILED) {
            reportClientException(runId, edgeId, status);
        }
    }

    @Override
    public void syncClientStatus(final long edgeId) {
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, mRunId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(EDGE_ID_ALIAS, edgeId);
            jsonObject.put(REPORT_STATUS, CLIENT_STATUS_MAP.get(mClientStatus));
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %s)", edgeId, MSG_MLOPS_CLIENT_STATUS_IDLE);
        }
        edgeCommunicator.sendMessage(FedMqttTopic.flclientStatus(edgeId), jsonObject.toString());
        edgeCommunicator.sendMessage(FedMqttTopic.STATUS, jsonObject.toString());
        edgeCommunicator.sendMessage(FedMqttTopic.RUN_STATUS, jsonObject.toString());
    }

    public void reportClientActiveStatus(final long edgeId) {
        JSONObject jsonObject = new JSONObject();
        if (mClientStatus != KEY_CLIENT_STATUS_OFFLINE &&
                mClientStatus != KEY_CLIENT_STATUS_IDLE &&
                mClientStatus != KEY_CLIENT_STATUS_FINISHED) {
            return;
        }
        try {
            notifyClientStatus(mRunId, KEY_CLIENT_STATUS_IDLE);
            jsonObject.put(EDGE_ID_ALIAS, edgeId);
            jsonObject.put(REPORT_STATUS, MSG_MLOPS_CLIENT_STATUS_IDLE);
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %s)", edgeId, MSG_MLOPS_CLIENT_STATUS_IDLE);
        }
        edgeCommunicator.sendMessage(FedMqttTopic.FL_CLIENT_ACTIVE, jsonObject.toString());
    }

    public void reportClientModelInfo(final long runId, final long edgeId, final int clientRound, final String model) {
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(ROUND_IDX, clientRound);
            jsonObject.put(CLIENT_MODEL_ADDRESS, model);
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %s, %s)", edgeId, clientRound, model);
        }
        edgeCommunicator.sendMessage(FedMqttTopic.CLIENT_MODEL, jsonObject.toString());
    }

    public void reportTrainingMetric(long edgeId, long runId, float accuracy, float loss) {
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("run_id", runId);
            jsonObject.put("edge_id", edgeId);
            jsonObject.put("accuracy", accuracy);
            jsonObject.put("loss", loss);
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingMetric(%s)", edgeId);
        }
        edgeCommunicator.sendMessage(FedMqttTopic.TRAINING_PROGRESS_AND_EVAL, jsonObject.toString());
    }

    public void reportSystemMetric(final long runId, final long edgeId) {
        JSONObject jsonObject = new JSONObject();
        final SysStats sysStats = SysStats.getInstance();
        try {
            jsonObject.put("run_id", runId);
            jsonObject.put("edge_id", edgeId);
            Float cpuUtilization = sysStats.getCpuUtilization();
            if (null != cpuUtilization) {
                jsonObject.put("cpu_utilization", cpuUtilization);
            }

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
        edgeCommunicator.sendMessage(FedMqttTopic.SYSTEM_PERFORMANCE, jsonObject.toString());
    }

    private void notifyClientStatus(final long runId, final int status) {
        LogHelper.d("FedMLDebug. notifyClientStatus [%s]", CLIENT_STATUS_MAP.get(status));
        mClientStatus = status;
        if (status == KEY_CLIENT_STATUS_IDLE || status == KEY_CLIENT_STATUS_KILLED ||
                status == KEY_CLIENT_STATUS_FINISHED || status == KEY_CLIENT_STATUS_FAILED) {
            mRunId = 0;
        } else {
            mRunId = runId;
        }
        if (mOnTrainingStatusListener != null) {
            mOnTrainingStatusListener.onStatusChanged(status);
        }
    }

    public void reportClientException(final long runId, final long edgeId, int status) {
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put(RUN_ID, runId);
            jsonObject.put(EDGE_ID, edgeId);
            jsonObject.put(REPORT_STATUS, CLIENT_STATUS_MAP.get(status));
        } catch (JSONException e) {
            LogHelper.e(e, "reportTrainingStatus(%d, %d)", edgeId, status);
        }
        edgeCommunicator.sendMessage(FedMqttTopic.exitTrainWithException(runId), jsonObject.toString());
    }
}
