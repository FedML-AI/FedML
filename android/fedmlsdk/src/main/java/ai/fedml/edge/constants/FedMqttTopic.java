package ai.fedml.edge.constants;

public final class FedMqttTopic {

    // Topic definition: "from/to/info"
    public static final String SYSTEM_PERFORMANCE = "fl_client/mlops/system_performance";

    public static final String DEVICE_INFO = "fl_client/mlops/device_info";

    // TODO: change to "fl_client/mlops/events"? (server may also need to use this topic)
    public static final String EVENT = "mlops/events";

    // edge status
    public static final String STATUS = "fl_client/mlops/status";

    // TODO: change to "fl_client/mlops/run_status"
    public static final String RUN_STATUS = "fl_run/fl_client/mlops/status";

    // response to MLOps's query of the client status
    // TODO: change to "flclient_agent/mlops/active"
    public static final String FL_CLIENT_ACTIVE = "flclient_agent/active";

    // TODO: change to "fl_client/mlops/client_model"
    public static final String CLIENT_MODEL = "fl_server/mlops/client_model";

    public static final String TRAINING_PROGRESS_AND_EVAL = "fl_client/mlops/training_progress_and_eval";

    // TODO: change to "flclient_agent/mlops/last_will_msg"
    public static final String MQTT_LAST_WILL_TOPIC = "flclient_agent/last_will_msg";

    // TODO: change to "flclient_agent/flserver_agent/$run_id/client_exit_train_with_exception"
    public static String exitTrainWithException(final long runId) {
        return "flserver_agent/" + runId + "/client_exit_train_with_exception";
    }

    public static String flclientStatus(final long edgeId) {
        return "fl_client/mlops/" + edgeId + "/status";
    }

    public static String modelUpload(final long runId, final long edgeId) {
        return "fedml_" + runId + "_" + edgeId;
    }

    public static String online(final long runId, final long edgeId) {
        return "fedml_" + runId + "_" + edgeId;
    }
}
