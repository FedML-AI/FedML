class MyMessage(object):
    """
    message type definition
    """

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_GRADIENT = 2

    # client to server
    MSG_TYPE_C2S_LOGITS = 3

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_TRAIN_LOGITS = "train_logits"
    MSG_ARG_KEY_TEST_LOGITS = "test_logits"
    MSG_ARG_KEY_GRADIENT = "gradient"
