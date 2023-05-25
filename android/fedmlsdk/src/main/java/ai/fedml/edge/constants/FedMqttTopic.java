package ai.fedml.edge.constants;

public class FedMqttTopic {

    public static final String SYSTEM_PERFORMANCE = "fl_client/mlops/system_performance";

    public static final String DEVICE_INFO = "fl_client/mlops/device_info";

    public static final String EVENT = "mlops/events";

    public static final String STATUS = "fl_client/mlops/status";

    public static final String RUN_STATUS = "fl_run/fl_client/mlops/status";

    public static final String FL_CLIENT_ACTIVE = "flclient_agent/active";

    public static final String CLIENT_MODEL = "fl_server/mlops/client_model";

    public static final String TRAINING_PROGRESS_AND_EVAL = "fl_client/mlops/training_progress_and_eval";

    public static String exitTrainWithException(final long runId) {
        return "flserver_agent/" + runId + "/client_exit_train_with_exception";
    }
}