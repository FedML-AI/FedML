class MyMessage(object):
    """
    message type definition
    """

    # cloud to edge
    MSG_TYPE_C2E_INIT_CONFIG = 1
    MSG_TYPE_C2E_SYNC_MODEL_TO_EDGE = 2

    # edge to cloud
    MSG_TYPE_E2C_SEND_MODEL_TO_CLOUD = 3
    MSG_TYPE_E2C_SEND_STATS_TO_CLOUD = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_MODEL_PARAMS_LIST = "model_params_list"
    MSG_ARG_KEY_EDGE_INDEX = "edge_idx"
    MSG_ARG_KEY_TOTAL_EDGE_CLIENTS = "total_edge_clients"
    MSG_ARG_KEY_SAMPLED_EDGE_CLIENTS = "sampled_edge_clients"
    MSG_ARG_KEY_TOTAL_SAMPLED_DATA_SIZE = "total_sampled_data_size"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"
