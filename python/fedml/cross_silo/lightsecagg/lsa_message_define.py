class MyMessage(object):
    """
    LightSecAgg Protocol:
       1 (server initializes the model parameters)
    -> 5 (clients send encoded mask to other clients via the server)
    -> 2 (the server transfers the encoded mask to clients)
    ==========the client is doing the model training=========
    -> 6 (send the trained model to the server)
    -> 4 (the server asks the active users to upload the aggregate mask)
    -> 7 (clients send the aggregate mask to the server)
    ==========          model aggregation          =========
    -> 3 (the server send the aggregated model to all clients)
    """

    # connection info
    MSG_TYPE_CONNECTION_IS_READY = 0

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_ENCODED_MASK_TO_CLIENT = 2
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 3
    MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT = 4
    MSG_TYPE_S2C_CHECK_CLIENT_STATUS = 9
    MSG_TYPE_S2C_FINISH = 10

    # client to server
    MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER = 5
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 6
    MSG_TYPE_C2S_SEND_MASK_TO_SERVER = 7
    MSG_TYPE_C2S_CLIENT_STATUS = 8

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_MODEL_PARAMS_URL = "model_params_url"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_ENCODED_MASK = "encoded_mask"
    MSG_ARG_KEY_ACTIVE_CLIENTS = "active_clinets"
    MSG_ARG_KEY_AGGREGATE_ENCODED_MASK = "aggregate_encoded_mask"
    MSG_ARG_KEY_CLIENT_ID = "client_id"

    MSG_ARG_KEY_CLIENT_STATUS = "client_status"
    MSG_ARG_KEY_CLIENT_OS = "client_os"

    MSG_ARG_KEY_EVENT_NAME = "event_name"
    MSG_ARG_KEY_EVENT_VALUE = "event_value"
    MSG_ARG_KEY_EVENT_MSG = "event_msg"

    """
        MLOps related message 
    """
    # Client Status
    MSG_MLOPS_CLIENT_STATUS_IDLE = "IDLE"
    MSG_MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MLOPS_CLIENT_STATUS_TRAINING = "TRAINING"
    MSG_MLOPS_CLIENT_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_CLIENT_STATUS_FINISHED = "FINISHED"

    # Server Status
    MSG_MLOPS_SERVER_STATUS_IDLE = "IDLE"
    MSG_MLOPS_SERVER_STATUS_STARTING = "STARTING"
    MSG_MLOPS_SERVER_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_SERVER_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_SERVER_STATUS_KILLED = "KILLED"
    MSG_MLOPS_SERVER_STATUS_FAILED = "FAILED"
    MSG_MLOPS_SERVER_STATUS_FINISHED = "FINISHED"

    # Client OS
    MSG_CLIENT_OS_ANDROID = "android"
    MSG_CLIENT_OS_IOS = "iOS"
    MSG_CLIENT_OS_Linux = "linux"
