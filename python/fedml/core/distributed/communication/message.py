import json
import sys


class Message(object):
    """
    A class for representing and working with messages in a communication system.
    """

    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_MODEL_PARAMS_URL = "model_params_url"
    MSG_ARG_KEY_MODEL_PARAMS_KEY = "model_params_key"

    def __init__(self, type="default", sender_id=0, receiver_id=0):
        """
        Initialize a Message instance.

        Args:
            type (str): The type of the message.
            sender_id (int): The ID of the sender.
            receiver_id (int): The ID of the receiver.
        """
        self.type = str(type)
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.msg_params = {}
        self.msg_params[Message.MSG_ARG_KEY_TYPE] = type
        self.msg_params[Message.MSG_ARG_KEY_SENDER] = sender_id
        self.msg_params[Message.MSG_ARG_KEY_RECEIVER] = receiver_id

    def init(self, msg_params):
        """
        Initialize the message with the provided message parameters.

        Args:
            msg_params (dict): A dictionary of message parameters.
        """
        self.msg_params = msg_params

    def init_from_json_string(self, json_string):
        """
        Initialize the message from a JSON string.

        Args:
            json_string (str): A JSON string representing the message.
        """
        self.msg_params = json.loads(json_string)
        self.type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[Message.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[Message.MSG_ARG_KEY_RECEIVER]

    def init_from_json_object(self, json_object):
        """
        Initialize the message from a JSON object.

        Args:
            json_object (dict): A JSON object representing the message.
        """
        self.msg_params = json_object
        self.type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[Message.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[Message.MSG_ARG_KEY_RECEIVER]

    def get_sender_id(self):
        """
        Get the ID of the sender.

        Returns:
            int: The sender's ID.
        """
        return self.sender_id

    def get_receiver_id(self):
        """
        Get the ID of the receiver.

        Returns:
            int: The receiver's ID.
        """
        return self.receiver_id

    def add_params(self, key, value):
        """
        Add a parameter to the message.

        Args:
            key (str): The key of the parameter.
            value (any): The value of the parameter.
        """
        self.msg_params[key] = value

    def get_params(self):
        """
        Get all the parameters of the message.

        Returns:
            dict: A dictionary of message parameters.
        """
        return self.msg_params

    def add(self, key, value):
        """
        Add a parameter to the message (alias for add_params).

        Args:
            key (str): The key of the parameter.
            value (any): The value of the parameter.
        """
        self.msg_params[key] = value

    def get(self, key):
        """
        Get the value of a parameter by its key.

        Args:
            key (str): The key of the parameter.

        Returns:
            any: The value of the parameter or None if not found.
        """
        if key not in self.msg_params.keys():
            return None
        return self.msg_params[key]

    def get_type(self):
        """
        Get the type of the message.

        Returns:
            str: The type of the message.
        """
        return self.msg_params[Message.MSG_ARG_KEY_TYPE]

    def to_string(self):
        """
        Convert the message to a string representation.

        Returns:
            dict: A dictionary representing the message.
        """
        return self.msg_params

    def to_json(self):
        """
        Serialize the message to a JSON string.

        Returns:
            str: A JSON string representing the message.
        """
        json_string = json.dumps(self.msg_params)
        print("json string size = " + str(sys.getsizeof(json_string)))
        return json_string

    def get_content(self):
        """
        Get a human-readable representation of the message.

        Returns:
            str: A string representing the message content.
        """
        print_dict = self.msg_params.copy()
        msg_str = str(self.__to_msg_type_string()) + ": " + str(print_dict)
        return msg_str

    def __to_msg_type_string(self):
        """
        Get a string representation of the message type.

        Returns:
            str: A string representing the message type.
        """
        type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        return type
