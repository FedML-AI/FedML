import argparse
import json
import logging
import time
import traceback
import uuid

from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.computing.scheduler.model_scheduler.modelops_configs import ModelOpsConfigs
from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager


class FedMLModelMetrics:
    def __init__(self, end_point_id, end_point_name, model_id, model_name, model_version,
                 infer_url, redis_addr, redis_port, redis_password, version="release"):
        self.redis_addr = redis_addr
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.config_version = version
        self.current_end_point_id = end_point_id
        self.current_end_point_name = end_point_name
        self.current_model_id = model_id
        self.current_model_name = model_name
        self.current_model_version = model_version
        self.current_infer_url = infer_url
        self.start_time = time.time_ns()
        self.monitor_mqtt_mgr = None
        self.ms_per_sec = 1000
        self.ns_per_ms = 1000 * 1000

    def set_start_time(self, start_time):
        if start_time is None:
            self.start_time = time.time_ns()
        else:
            self.start_time = start_time

    def calc_metrics(self, end_point_id, end_point_name,
                     model_id, model_name, model_version,
                     inference_output_url, device_id):
        total_latency, avg_latency, total_request_num, current_qps, timestamp = 0, 0, 0, 0, 0
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        metrics_item = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_latest_monitor_metrics(end_point_id, end_point_name, model_name, model_version)
        logging.info(f"Calculated metrics_item: {metrics_item}")
        if metrics_item is not None:
            total_latency, avg_latency, total_request_num, current_qps, avg_qps, timestamp, _ = \
                FedMLModelCache.get_instance(self.redis_addr, self.redis_port).get_metrics_item_info(metrics_item)
        cost_time = (time.time_ns() - self.start_time) / self.ns_per_ms
        total_latency += cost_time
        total_request_num += 1
        current_qps = 1 / (cost_time / self.ms_per_sec)
        current_qps = format(current_qps, '.6f')
        avg_qps = total_request_num * 1.0 / (total_latency / self.ms_per_sec)
        avg_qps = format(avg_qps, '.6f')
        avg_latency = format(total_latency / total_request_num / self.ms_per_sec, '.6f')

        timestamp = int(format(time.time_ns()/1000.0, '.0f'))
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port).set_monitor_metrics(end_point_id,
                                                                                           end_point_name,
                                                                                           model_name,
                                                                                           model_version,
                                                                                           total_latency,
                                                                                           avg_latency,
                                                                                           total_request_num,
                                                                                           current_qps, avg_qps,
                                                                                           timestamp,
                                                                                           str(device_id))

    def start_monitoring_metrics_center(self):
        self.build_metrics_report_channel()

    def build_metrics_report_channel(self):
        args = {"config_version": "release"}
        mqtt_config, _ = ModelOpsConfigs.get_instance(args).fetch_configs(self.config_version)
        self.monitor_mqtt_mgr = MqttManager(
            mqtt_config["BROKER_HOST"],
            mqtt_config["BROKER_PORT"],
            mqtt_config["MQTT_USER"],
            mqtt_config["MQTT_PWD"],
            mqtt_config["MQTT_KEEPALIVE"],
            "FedML_ModelMonitor_" + str(uuid.uuid4())
        )
        self.monitor_mqtt_mgr.add_connected_listener(self.on_mqtt_connected)
        self.monitor_mqtt_mgr.add_disconnected_listener(self.on_mqtt_disconnected)
        self.monitor_mqtt_mgr.connect()
        self.monitor_mqtt_mgr.loop_start()

        index = 0
        while True:
            # Frequency of sending monitoring metrics
            time.sleep(5)
            try:
                index = self.send_monitoring_metrics(index)
            except Exception as e:
                logging.info("Exception when processing monitoring metrics: {}".format(traceback.format_exc()))

        self.monitor_mqtt_mgr.loop_stop()
        self.monitor_mqtt_mgr.disconnect()

    def send_monitoring_metrics(self, index):
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        metrics_item, inc_index = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_monitor_metrics_item(self.current_end_point_id, self.current_end_point_name,
                                     self.current_model_name, self.current_model_version, index)
        if metrics_item is None:
            return index
        total_latency, avg_latency, total_request_num, current_qps, avg_qps, timestamp, device_id = \
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port).get_metrics_item_info(metrics_item)
        deployment_monitoring_topic_prefix = "model_ops/model_device/return_inference_monitoring"
        deployment_monitoring_topic = "{}/{}".format(deployment_monitoring_topic_prefix, self.current_end_point_id)
        deployment_monitoring_payload = {"model_name": self.current_model_name,
                                         "model_id": self.current_model_id,
                                         "model_url": self.current_infer_url,
                                         "end_point_id": self.current_end_point_id,
                                         "latency": float(avg_latency),
                                         "qps": float(avg_qps),
                                         "total_request_num": int(total_request_num),
                                         "timestamp": timestamp,
                                         "edgeId": device_id}
        # logging.info("send monitor metrics {}".format(json.dumps(deployment_monitoring_payload)))

        self.monitor_mqtt_mgr.send_message_json(deployment_monitoring_topic, json.dumps(deployment_monitoring_payload))
        self.monitor_mqtt_mgr.send_message_json(deployment_monitoring_topic_prefix,
                                                json.dumps(deployment_monitoring_payload))
        return inc_index

    def on_mqtt_connected(self, mqtt_client_object):
        pass

    def on_mqtt_disconnected(self, mqtt_client_object):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", "-v", type=str, default="release", help="version")
    parser.add_argument("--end_point_id", "-ep", help="end point id")
    parser.add_argument("--end_point_name", "-epn", help="end point name")
    parser.add_argument("--model_id", "-mi", type=str, help='model id')
    parser.add_argument("--model_name", "-mn", type=str, help="model name")
    parser.add_argument("--model_version", "-mv", type=str, help="model model_version")
    parser.add_argument("--infer_url", "-iu", type=str, help="inference url")
    parser.add_argument("--redis_addr", "-ra", type=str, default="local")
    parser.add_argument("--redis_port", "-rp", type=str, default="6379")
    parser.add_argument("--redis_password", "-rpw", type=str, default="fedml_default")
    args = parser.parse_args()

    monitor_center = FedMLModelMetrics(args.end_point_id, args.end_point_name,
                                       args.model_id, args.model_name, args.model_version,
                                       args.infer_url,
                                       args.redis_addr, args.redis_port, args.redis_password,
                                       version=args.version)
    monitor_center.start_monitoring_metrics_center()

