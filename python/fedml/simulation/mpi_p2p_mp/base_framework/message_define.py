class MyMessage(object):
    """
    message type definition
    """

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_INFORMATION = 2

    # client to server
    MSG_TYPE_C2S_INFORMATION = 3

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_INFORMATION = "information"
