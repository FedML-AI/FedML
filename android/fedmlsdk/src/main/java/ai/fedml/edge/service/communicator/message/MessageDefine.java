package ai.fedml.edge.service.communicator.message;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public interface MessageDefine {
    int MSG_TYPE_CONNECTION_IS_READY = 0;
    int MSG_TYPE_S2C_INIT_CONFIG = 1;
    int MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2;
    int MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3;
    int MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4;
    int MSG_TYPE_C2S_CLIENT_STATUS = 5;
    int MSG_TYPE_S2C_CHECK_CLIENT_STATUS = 6;

    int MSG_TYPE_S2C_FINISH = 7;

    String TOPIC = "topic";
    String GROUP_ID = "groupid";
    String RUN_ID = "run_id";
    String EDGE_ID = "edge_id";
    String ROUND_IDX = "round_idx";
    String CLIENT_MODEL_ADDRESS = "client_model_s3_address";

    String MSG_TYPE = "msg_type";
    String MSG_ARG_KEY_SENDER = "sender";
    String MSG_ARG_KEY_RECEIVER = "receiver";

    // message payload keywords definition
    String MSG_ARG_KEY_NUM_SAMPLES = "num_samples";
    String MSG_ARG_KEY_MODEL_PARAMS = "model_params";
    String MSG_ARG_KEY_CLIENT_INDEX = "client_idx";

    String MSG_ARG_KEY_TRAIN_CORRECT = "train_correct";
    String MSG_ARG_KEY_TRAIN_ERROR = "train_error";
    String MSG_ARG_KEY_TRAIN_NUM = "train_num_sample";

    String MSG_ARG_KEY_TEST_CORRECT = "test_correct";
    String MSG_ARG_KEY_TEST_ERROR = "test_error";
    String MSG_ARG_KEY_TEST_NUM = "test_num_sample";

    String MSG_ARG_KEY_CLIENT_STATUS = "client_status";
    String MSG_ARG_KEY_CLIENT_OS = "client_os";


    // Client Status
    String MQTT_LAST_WILL_TOPIC = "flclient_agent/last_will_msg";
    String MQTT_REPORT_ACTIVE_STATUS_TOPIC = "flclient_agent/active";

    String MSG_MLOPS_CLIENT_STATUS_OFFLINE = "OFFLINE";
    String MSG_MLOPS_CLIENT_STATUS_IDLE = "IDLE";
    String MSG_MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING";
    String MSG_MLOPS_CLIENT_STATUS_PENDING = "PENDING";
    String MSG_MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING";
    String MSG_MLOPS_CLIENT_STATUS_TRAINING = "TRAINING";
    String MSG_MLOPS_CLIENT_STATUS_KILLED = "KILLED";
    String MSG_MLOPS_CLIENT_STATUS_FINISHED = "FINISHED";

    String MSG_MLOPS_CLIENT_STATUS_FAILED = "FAILED";

    int KEY_CLIENT_STATUS_OFFLINE = -1;
    int KEY_CLIENT_STATUS_IDLE = 0;
    int KEY_CLIENT_STATUS_UPGRADING = 1;
    int KEY_CLIENT_STATUS_PENDING = 2;
    int KEY_CLIENT_STATUS_INITIALIZING = 3;
    int KEY_CLIENT_STATUS_TRAINING = 5;
    int KEY_CLIENT_STATUS_KILLED = 6;
    int KEY_CLIENT_STATUS_FINISHED = 7;

    int KEY_CLIENT_STATUS_FAILED = 8;

    Map<Integer, String> CLIENT_STATUS_MAP = Collections.unmodifiableMap(new HashMap<Integer, String>() {
        {
            put(KEY_CLIENT_STATUS_OFFLINE, MSG_MLOPS_CLIENT_STATUS_OFFLINE);
            put(KEY_CLIENT_STATUS_IDLE, MSG_MLOPS_CLIENT_STATUS_IDLE);
            put(KEY_CLIENT_STATUS_UPGRADING, MSG_MLOPS_CLIENT_STATUS_UPGRADING);
            put(KEY_CLIENT_STATUS_PENDING, MSG_MLOPS_CLIENT_STATUS_PENDING);
            put(KEY_CLIENT_STATUS_INITIALIZING, MSG_MLOPS_CLIENT_STATUS_INITIALIZING);
            put(KEY_CLIENT_STATUS_TRAINING, MSG_MLOPS_CLIENT_STATUS_TRAINING);
            put(KEY_CLIENT_STATUS_KILLED, MSG_MLOPS_CLIENT_STATUS_KILLED);
            put(KEY_CLIENT_STATUS_FINISHED, MSG_MLOPS_CLIENT_STATUS_FINISHED);
            put(KEY_CLIENT_STATUS_FAILED, MSG_MLOPS_CLIENT_STATUS_FAILED);
        }
    });


    // Client OS
    String MSG_CLIENT_OS_ANDROID = "Android";

    // report
    String REPORT_STATUS = "status";
    // run_config
    String RUN_CONFIG = "run_config";

    // hyper parameters config
    String HYPER_PARAMETERS_CONFIG = "parameters";
    // train_args
    String TRAIN_ARGS = "train_args";
    // comm_round
    String COMM_ROUND = "comm_round";
    // data_args
    String DATA_ARGS = "data_args";
    // DATASET_TYPE
    String DATASET_TYPE = "dataset";
    String TRAIN_ARGS_BATCH_SIZE = "batch_size";
    String TRAIN_ARGS_LR = "learning_rate";
    String TRAIN_SERVER_ID = "server_id";
    String TRAIN_ARGS_EPOCH_NUM = "epochs";
    String DATA_ARGS_TRAIN_SIZE = "train_size";
    String DATA_ARGS_TEST_SIZE = "test_size";

}
