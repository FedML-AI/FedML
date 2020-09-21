class MyMessage(object):
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_ARCH_PARAMS = "arch_params"
    MSG_ARG_KEY_LOCAL_TRAINING_ACC = "local_training_acc"
    MSG_ARG_KEY_LOCAL_TRAINING_LOSS = "local_training_loss"
    MSG_ARG_KEY_LOCAL_TEST_ACC = "local_test_acc"
    MSG_ARG_KEY_LOCAL_TEST_LOSS = "local_test_loss"
