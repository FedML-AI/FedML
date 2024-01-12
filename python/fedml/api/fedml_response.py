from enum import Enum


class ResponseCode(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ERROR = "ERROR"


class FedMLResponse(object):

    def __init__(self, code: ResponseCode, message: str, data = None):
        self.code = code
        self.message = message
        self.data = data
