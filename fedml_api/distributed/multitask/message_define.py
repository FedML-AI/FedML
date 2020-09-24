class MyMessage(object):
    """
        message type definition
    """
    # message to neighbor
    MSG_TYPE_INIT = 1
    MSG_TYPE_SEND_MSG_TO_NEIGHBOR = 2
    MSG_TYPE_METRICS = 3

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_LOCAL_TRAIN_ACC = "train_acc"
    MSG_ARG_KEY_LOCAL_TRAIN_LOSS = "train_loss"

    MSG_ARG_KEY_LOCAL_TEST_ACC = "test_acc"
    MSG_ARG_KEY_LOCAL_TEST_LOSS = "test_loss"
