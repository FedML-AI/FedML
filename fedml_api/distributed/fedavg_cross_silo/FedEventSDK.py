import os
import sys
import threading

import json
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from fedml_core.distributed.communication.mqtt_s3.mqtt_s3_status_manager import MqttS3StatusManager


class FedEventSDK:
    EVENT_TYPE_STARTED = 0
    EVENT_TYPE_ENDED = 1

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(FedEventSDK, "_instance"):
            with FedEventSDK._instance_lock:
                if not hasattr(FedEventSDK, "_instance"):
                    FedEventSDK._instance = object.__new__(cls)
        return FedEventSDK._instance

    def __init__(self, args):
        self.args = args
        self.run_id = args.run_id
        if args.silo_rank == 0:
            self.edge_id = 0
        else:
            self.edge_id = json.loads(args.client_ids)[0]
        self.com_manager = MqttS3StatusManager(
            args.mqtt_config_path, args.s3_config_path, topic=args.run_id)

    def log_event_started(self, event_name, event_value=None, event_edge_id=None):
        if event_value is None:
            event_value_passed = ""
        else:
            event_value_passed = event_value

        if event_edge_id is not None:
            edge_id = event_edge_id
        else:
            edge_id = self.edge_id

        event_topic, event_msg = self.__build_event_mqtt_msg(self.args.run_id, edge_id,
                                                             FedEventSDK.EVENT_TYPE_STARTED,
                                                             event_name, event_value_passed)
        self.com_manager.send_message_json(event_topic, json.dumps(event_msg))

    def log_event_ended(self, event_name, event_value=None, event_edge_id=None):
        if event_value is None:
            event_value_passed = ""
        else:
            event_value_passed = event_value

        if event_edge_id is not None:
            edge_id = event_edge_id
        else:
            edge_id = self.edge_id

        event_topic, event_msg = self.__build_event_mqtt_msg(self.args.run_id, edge_id,
                                                             FedEventSDK.EVENT_TYPE_ENDED,
                                                             event_name, event_value_passed)
        self.com_manager.send_message_json(event_topic, json.dumps(event_msg))

    @staticmethod
    def __build_event_mqtt_msg(run_id, edge_id, event_type, event_name, event_value):
        event_topic = "/mlops/events"
        event_msg = {}
        if event_type == FedEventSDK.EVENT_TYPE_STARTED:
            event_msg = {"run_id": run_id, "edge_id": edge_id, "event_name": event_name,
                         "event_value": event_value, "started_time": int(time.time())}
        elif event_type == FedEventSDK.EVENT_TYPE_ENDED:
            event_msg = {"run_id": run_id, "edge_id": edge_id, "event_name": event_name,
                         "event_value": event_value, "ended_time": int(time.time())}

        return event_topic, event_msg
