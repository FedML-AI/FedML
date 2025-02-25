import json
import time


class FedMLMessageEntity(object):
    def __init__(self, topic=None, payload=None, run_id=None, device_id=None, message_body: dict = None):
        self.topic = topic
        self.payload = payload
        self.run_id = run_id
        self.device_id = device_id
        if message_body is not None:
            self.from_message_body(message_body=message_body)

    def from_message_body(self, message_body: dict = None):
        self.topic = message_body.get("topic", None)
        self.payload = message_body.get("payload", None)
        if self.payload is not None:
            payload_json = json.loads(self.payload)
            self.run_id = payload_json.get("run_id", None)
            self.run_id = payload_json.get("runId", None) if self.run_id is None else self.run_id
            self.device_id = payload_json.get("edge_id", None)
            self.device_id = payload_json.get("ID", None) if self.device_id is None else self.device_id

    def get_message_body(self):
        message_body = {"topic": self.topic, "payload": self.payload, "run_id": self.run_id}
        return message_body


class FedMLMessageRecord(object):
    def __init__(self, message_id=None, message_body=None, json_record=None):
        self.message_id = message_id
        self.message_body = message_body
        self.timestamp = time.time_ns() / 1000.0 / 1000.0
        if json_record is not None:
            self.from_message_record(json_record=json_record)

    def get_json_record(self):
        return {"message_id": self.message_id, "message_body": self.message_body, "timestamp": self.timestamp}

    def from_message_record(self, json_record: dict = None):
        self.message_id = json_record.get("message_id", None)
        self.message_body = json_record.get("message_body", None)
        self.timestamp = json_record.get("timestamp", None)


class FedMLStatusEntity(object):
    def __init__(self, topic=None, payload=None, status_msg_body: dict = None):
        self.topic = topic
        self.payload = payload
        self.run_id = None
        self.edge_id = None
        self.server_id = None
        self.status = None
        if status_msg_body is not None:
            self.from_message_body(status_msg_body=status_msg_body)

    def from_message_body(self, status_msg_body: dict = None):
        self.topic = status_msg_body.get("topic", None)
        self.payload = status_msg_body.get("payload", None)
        if self.payload is not None:
            payload_json = json.loads(self.payload)
            self.run_id = payload_json.get("run_id", None)
            self.run_id = payload_json.get("runId", None) if self.run_id is None else self.run_id
            self.edge_id = payload_json.get("edge_id", None)
            self.server_id = payload_json.get("server_id", None)
            self.status = payload_json.get("status", None)

    def get_message_body(self):
        status_msg_body = {"topic": self.topic, "payload": self.payload, "run_id": self.run_id}
        return status_msg_body


class LogArgs:
    def __init__(self, role=None, edge_id=None, server_id=None, log_server_url=None, log_file_dir=None):
        self.role = role
        self.edge_id = edge_id
        self.server_id = server_id
        self.log_server_url = log_server_url
        self.log_file_dir = log_file_dir
