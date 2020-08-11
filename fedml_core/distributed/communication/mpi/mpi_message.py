class MPIMessage(object):
    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    def __init__(self):
        self._message_dict = {}
        return

    def init(self, message_dict):
        self._message_dict = message_dict

    def add(self, key, value):
        self._message_dict[key] = value

    def get(self, key):
        return self._message_dict[key]

    def get_type(self):
        return self._message_dict[MPIMessage.MSG_ARG_KEY_TYPE]

    def to_string(self):
        return self._message_dict

    def get_content(self):
        print_dict = self._message_dict.copy()
        msg_str = str(self.__to_msg_type_string()) + ": " + str(print_dict)
        return msg_str

    def __to_msg_type_string(self):
        type = self._message_dict[MPIMessage.MSG_ARG_KEY_TYPE]
        return type
