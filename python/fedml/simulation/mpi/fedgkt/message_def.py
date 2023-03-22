class MyMessage(object):
    """
    message type definition
    """

    # client-oriented message
    MSG_TYPE_C2S_SEND_FEATURE_AND_LOGITS = 3

    # server-oriented message
    MSG_TYPE_S2C_INIT_CONFIG = 1  # send
    MSG_TYPE_S2C_SYNC_TO_CLIENT = 2  # send

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_FEATURE = "feature"
    MSG_ARG_KEY_LOGITS = "logits"
    MSG_ARG_KEY_LABELS = "labels"
    MSG_ARG_KEY_FEATURE_TEST = "feature_test"
    MSG_ARG_KEY_LABELS_TEST = "labels_test"
    MSG_ARG_KEY_GLOBAL_LOGITS = "global_logits"
