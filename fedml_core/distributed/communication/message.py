class Message(object):

    def __init__(self, type, sender_id, receiver_id):
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.msg_params = dict()

    def get_type(self):
        return self.type

    def get_sender_id(self):
        return self.sender_id

    def get_receiver_id(self):
        return self.receiver_id

    def add_params(self, key, value):
        self.msg_params[key] = value

    def get_params(self):
        return self.msg_params