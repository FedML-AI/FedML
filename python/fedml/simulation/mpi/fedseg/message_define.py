class MyMessage(object):
    """
    message type definition
    """

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"
    MSG_ARG_KEY_TRAIN_EVALUATION_METRICS = "train_evaluation_metrics"
    MSG_ARG_KEY_TEST_EVALUATION_METRICS = "test_evaluation_metrics"
