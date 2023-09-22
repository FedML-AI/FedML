import json
import logging
import threading
import time

from .mlops_utils import MLOpsUtils


class MLOpsProfilerEvent:
    EVENT_TYPE_STARTED = 0
    EVENT_TYPE_ENDED = 1

    _instance_lock = threading.Lock()
    _sys_perf_profiling = False
    _enable_wandb = False

    def __new__(cls, *args, **kwargs):
        if not hasattr(MLOpsProfilerEvent, "_instance"):
            with MLOpsProfilerEvent._instance_lock:
                if not hasattr(MLOpsProfilerEvent, "_instance"):
                    MLOpsProfilerEvent._instance = object.__new__(cls)
        return MLOpsProfilerEvent._instance

    def __init__(self, args):
        """
        Initialize the MLOpsProfilerEvent.

        Args:
            args: The system arguments containing configuration settings.
        """
        self.args = args
        if args is not None and hasattr(args, "enable_wandb") and args.enable_wandb is not None:
            self.enable_wandb = args.enable_wandb
        else:
            self.enable_wandb = False

        self.edge_id = 0
        self.com_manager = None
        if args is not None:
            self.run_id = args.run_id
            self.set_messenger(self.com_manager, args)
        else:
            self.run_id = 0

    def set_messenger(self, msg_messenger, args=None):
        """
        Set the messenger for communication.

        Args:
            msg_messenger: The messenger for communication.
            args: The system arguments containing configuration settings.
        """
        self.com_manager = msg_messenger
        if args is None:
            return
        if args.rank == 0:
            if hasattr(args, "server_id"):
                self.edge_id = args.server_id
            else:
                self.edge_id = 0
        else:
            if hasattr(args, "client_id"):
                self.edge_id = args.client_id
            elif hasattr(args, "client_id_list"):
                edge_ids = json.loads(args.client_id_list)
                if len(edge_ids) > 0:
                    self.edge_id = edge_ids[0]
                else:
                    self.edge_id = 0
            else:
                self.edge_id = 0

    @classmethod
    def enable_wandb_tracking(cls):
        """
        Enable W&B (Weights and Biases) tracking.
        """
        cls._enable_wandb = True

    @classmethod
    def enable_sys_perf_profiling(cls):
        """
        Enable system performance profiling.
        """
        cls._sys_perf_profiling = True

    @classmethod
    def log_to_wandb(cls, metric):
        """
        Log a metric to W&B (Weights and Biases).

        Args:
            metric: The metric to log.
        """
        if cls._enable_wandb:
            import wandb
            wandb.log(metric)

    def log_event_started(self, event_name, event_value=None, event_edge_id=None):
        """
        Log the start of an event.

        Args:
            event_name: The name of the event.
            event_value: The value associated with the event.
            event_edge_id: The ID of the edge device associated with the event.
        """
        if event_value is None:
            event_value_passed = ""
        else:
            event_value_passed = event_value

        if event_edge_id is not None:
            edge_id = event_edge_id
        else:
            edge_id = self.edge_id

        time.sleep(0.1)
        event_topic, event_msg = self.__build_event_mqtt_msg(
            self.args.run_id,
            edge_id,
            MLOpsProfilerEvent.EVENT_TYPE_STARTED,
            event_name,
            event_value_passed,
        )
        event_msg_str = json.dumps(event_msg)
        logging.info("Event started, {}".format(event_msg_str))
        self.com_manager.send_message_json(event_topic, event_msg_str)

    def log_event_ended(self, event_name, event_value=None, event_edge_id=None):
        """
        Log the end of an event.

        Args:
            event_name: The name of the event.
            event_value: The value associated with the event.
            event_edge_id: The ID of the edge device associated with the event.
        """
        if event_value is None:
            event_value_passed = ""
        else:
            event_value_passed = event_value

        if event_edge_id is not None:
            edge_id = event_edge_id
        else:
            edge_id = self.edge_id

        time.sleep(0.1)
        event_topic, event_msg = self.__build_event_mqtt_msg(
            self.args.run_id,
            edge_id,
            MLOpsProfilerEvent.EVENT_TYPE_ENDED,
            event_name,
            event_value_passed,
        )
        event_msg_str = json.dumps(event_msg)
        logging.info("Event ended, {}".format(event_msg_str))
        self.com_manager.send_message_json(event_topic, event_msg_str)
        self.com_manager.send_message_json(event_topic, event_msg_str)

    @staticmethod
    def __build_event_mqtt_msg(run_id, edge_id, event_type, event_name, event_value):
        """
        Build an MQTT message for an event.

        Args:
            run_id: The ID of the run.
            edge_id: The ID of the edge device.
            event_type: The type of the event (started or ended).
            event_name: The name of the event.
            event_value: The value associated with the event.

        Returns:
            event_topic: The MQTT topic for the event.
            event_msg: The MQTT message for the event.
        """
        event_topic = "mlops/events"
        event_msg = {}
        if event_type == MLOpsProfilerEvent.EVENT_TYPE_STARTED:
            current_time_ms = MLOpsUtils.get_ntp_time()
            if current_time_ms is None:
                current_time = int(time.time()*1000)
            else:
                current_time = int(current_time_ms)
            event_msg = {
                "run_id": run_id,
                "edge_id": edge_id,
                "event_name": event_name,
                "event_value": event_value,
                "started_time": int(current_time),
            }
        elif event_type == MLOpsProfilerEvent.EVENT_TYPE_ENDED:
            current_time_ms = MLOpsUtils.get_ntp_time()
            if current_time_ms is None:
                current_time = int(time.time()*1000)
            else:
                current_time = int(current_time_ms)
            event_msg = {
                "run_id": run_id,
                "edge_id": edge_id,
                "event_name": event_name,
                "event_value": event_value,
                "ended_time": int(current_time),
            }

        return event_topic, event_msg
