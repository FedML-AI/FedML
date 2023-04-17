export var MyMessage;
(function (MyMessage) {
  // message type definition"
  // connection info"
  MyMessage[MyMessage.MSG_TYPE_CONNECTION_IS_READY = 0] = 'MSG_TYPE_CONNECTION_IS_READY'
  // server to client
  MyMessage[MyMessage.MSG_TYPE_S2C_INIT_CONFIG = 1] = 'MSG_TYPE_S2C_INIT_CONFIG'
  MyMessage[MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2] = 'MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT'
  MyMessage[MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS = 6] = 'MSG_TYPE_S2C_CHECK_CLIENT_STATUS'
  MyMessage[MyMessage.MSG_TYPE_S2C_FINISH = 7] = 'MSG_TYPE_S2C_FINISH'
  // client to server
  MyMessage[MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3] = 'MSG_TYPE_C2S_SEND_MODEL_TO_SERVER'
  MyMessage[MyMessage.MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4] = 'MSG_TYPE_C2S_SEND_STATS_TO_SERVER'
  MyMessage[MyMessage.MSG_TYPE_C2S_CLIENT_STATUS = 5] = 'MSG_TYPE_C2S_CLIENT_STATUS'
  MyMessage.MSG_ARG_KEY_TYPE = 'msg_type'
  MyMessage.MSG_ARG_KEY_SENDER = 'sender'
  MyMessage.MSG_ARG_KEY_RECEIVER = 'receiver'
  // message payload keywords definition"
  MyMessage.MSG_ARG_KEY_NUM_SAMPLES = 'num_samples'
  MyMessage.MSG_ARG_KEY_MODEL_PARAMS = 'model_params'
  MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL = 'model_params_url'
  MyMessage.MSG_ARG_KEY_CLIENT_INDEX = 'client_idx'
  MyMessage.MSG_ARG_KEY_TRAIN_CORRECT = 'train_correct'
  MyMessage.MSG_ARG_KEY_TRAIN_ERROR = 'train_error'
  MyMessage.MSG_ARG_KEY_TRAIN_NUM = 'train_num_sample'
  MyMessage.MSG_ARG_KEY_TEST_CORRECT = 'test_correct'
  MyMessage.MSG_ARG_KEY_TEST_ERROR = 'test_error'
  MyMessage.MSG_ARG_KEY_TEST_NUM = 'test_num_sample'
  MyMessage.MSG_ARG_KEY_CLIENT_STATUS = 'client_status'
  MyMessage.MSG_ARG_KEY_CLIENT_OS = 'client_os'
  MyMessage.MSG_ARG_KEY_EVENT_NAME = 'event_name'
  MyMessage.MSG_ARG_KEY_EVENT_VALUE = 'event_value'
  MyMessage.MSG_ARG_KEY_EVENT_MSG = 'event_msg'
  // MLOps related message
  // Client Status
  MyMessage.MSG_MLOPS_CLIENT_STATUS_IDLE = 'IDLE'
  MyMessage.MSG_MLOPS_CLIENT_STATUS_UPGRADING = 'UPGRADING'
  MyMessage.MSG_MLOPS_CLIENT_STATUS_INITIALIZING = 'INITIALIZING'
  MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING = 'TRAINING'
  MyMessage.MSG_MLOPS_CLIENT_STATUS_FINISHED = 'FINISHED'
  // Server Status
  MyMessage.MSG_MLOPS_SERVER_STATUS_IDLE = 'IDLE'
  MyMessage.MSG_MLOPS_SERVER_STATUS_STARTING = 'STARTING'
  MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING = 'RUNNING'
  MyMessage.MSG_MLOPS_SERVER_STATUS_KILLED = 'KILLED'
  MyMessage.MSG_MLOPS_SERVER_STATUS_FAILED = 'FAILED'
  MyMessage.MSG_MLOPS_SERVER_STATUS_FINISHED = 'FINISHED'
  // Client OS
  MyMessage.MSG_CLIENT_OS_ANDROID = 'android'
  MyMessage.MSG_CLIENT_OS_IOS = 'iOS'
  MyMessage.MSG_CLIENT_OS_Linux = 'linux'
})(MyMessage || (MyMessage = {}))
