import json
import logging
import multiprocessing
import time

from .mlops_status import MLOpsStatus
from .system_stats import SysStats


class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance


class MLOpsMetrics(Singleton):
    def __init__(self):
        self.messenger = None
        self.args = None
        self.run_id = None
        self.edge_id = None
        self.server_agent_id = None
        self.sys_performances = None
        self.is_sys_perf_reporting = False

    def set_messenger(self, msg_messenger, args=None):
        self.messenger = msg_messenger
        if args is not None:
            self.args = args
            self.run_id = args.run_id
            if args.rank == 0:
                if hasattr(args, "server_id"):
                    self.edge_id = args.server_id
                else:
                    self.edge_id = 0
            else:
                client_id_list = json.loads(args.client_id_list)
                self.edge_id = client_id_list[0]

            if hasattr(args, "server_agent_id"):
                self.server_agent_id = args.server_agent_id
            else:
                self.server_agent_id = self.edge_id

    def report_client_training_status(self, edge_id, status):
        """
            this is used for notifying the client status to MLOps (both web UI, FedML CLI and backend can consume it)
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_training_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_status(edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)
        self.report_client_id_status(run_id, edge_id, status)

    def broadcast_client_training_status(self, edge_id, status):
        """
            this is used for broadcasting the client status to MLOps (both web UI and backend can consume it)
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_training_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_id_status(self, run_id, edge_id, status):
        """
            this is used for communication between client agent (FedML cli module) and client
        """
        topic_name = "fl_client/flclient_agent_" + str(edge_id) + "/status"
        msg = {"run_id": run_id, "edge_id": edge_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_id_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_agent_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_status(self, run_id, status):
        topic_name = "fl_server/mlops/status"
        msg = {"run_id": run_id, "edge_id": self.edge_id, "status": status}
        logging.info("report_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        MLOpsStatus.get_instance().set_server_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)
        self.report_server_id_status(run_id, status)

    def broadcast_server_training_status(self, run_id, status):
        topic_name = "fl_server/mlops/status"
        msg = {"run_id": run_id, "edge_id": self.edge_id, "status": status}
        logging.info("broadcast_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_id_status(self, run_id, status):
        server_agent_id = self.server_agent_id
        topic_name = "fl_server/flserver_agent_" + str(server_agent_id) + "/status"
        msg = {"run_id": run_id, "edge_id": self.edge_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_server_id_status server id {}".format(server_agent_id))
        logging.info("report_server_id_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_server_agent_status(server_agent_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_training_metric(self, metric_json):
        topic_name = "fl_client/mlops/training_metrics"
        logging.info("report_client_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_metric(self, metric_json):
        topic_name = "fl_server/mlops/training_progress_and_eval"
        logging.info("report_server_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_round_info(self, round_info):
        topic_name = "fl_server/mlops/training_roundx"
        logging.info(
            "report_server_training_round_info. message_json = %s" % round_info
        )
        message_json = json.dumps(round_info)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_model_info(self, model_info_json):
        topic_name = "fl_server/mlops/client_model"
        logging.info("report_client_model_info. message_json = %s" % model_info_json)
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_aggregated_model_info(self, model_info_json):
        topic_name = "fl_server/mlops/global_aggregated_model"
        logging.info(
            "report_aggregated_model_info. message_json = %s" % model_info_json
        )
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_system_metric(self, metric_json=None):
        topic_name = "fl_client/mlops/system_performance"
        if metric_json is None:
            if self.sys_performances is None:
                self.sys_performances = SysStats()
            if self.sys_performances is None:
                return

            self.sys_performances.produce_info()
            metric_json = {
                "run_id": self.run_id,
                "edge_id": self.edge_id,
                "cpu_utilization": round(
                    self.sys_performances.get_cpu_utilization(), 4
                ),
                "SystemMemoryUtilization": round(
                    self.sys_performances.get_system_memory_utilization(), 4
                ),
                "process_memory_in_use": round(
                    self.sys_performances.get_process_memory_in_use(), 4
                ),
                "process_memory_in_use_size": round(
                    self.sys_performances.get_process_memory_in_use_size(), 4
                ),
                "process_memory_available": round(
                    self.sys_performances.get_process_memory_available(), 4
                ),
                "process_cpu_threads_in_use": round(
                    self.sys_performances.get_process_cpu_threads_in_use(), 4
                ),
                "disk_utilization": round(
                    self.sys_performances.get_disk_utilization(), 4
                ),
                "network_traffic": round(
                    self.sys_performances.get_network_traffic(), 4
                ),
                "gpu_utilization": round(
                    self.sys_performances.get_gpu_utilization(), 4
                ),
                "gpu_temp": round(self.sys_performances.get_gpu_temp(), 4),
                "gpu_time_spent_accessing_memory": round(
                    self.sys_performances.get_gpu_time_spent_accessing_memory(), 4
                ),
                "gpu_memory_allocated": round(
                    self.sys_performances.get_gpu_memory_allocated(), 4
                ),
                "gpu_power_usage": round(
                    self.sys_performances.get_gpu_power_usage(), 4
                ),
            }
        logging.info("report_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_logs_updated(self, run_id):
        topic_name = "mlops/runtime_logs/" + str(run_id)
        msg = {"time": time.time()}
        message_json = json.dumps(msg)
        logging.info("report_logs_updated. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def set_sys_reporting_status(self, enable):
        self.is_sys_perf_reporting = enable

    def is_system_perf_reporting(self):
        return self.is_sys_perf_reporting

    @staticmethod
    def report_sys_perf():
        sys_stats_process = multiprocessing.Process(
            target=MLOpsMetrics._report_sys_performances
        )
        sys_stats_process.start()

    @staticmethod
    def _report_sys_performances():
        # mlops_metrics is a single instance
        mlops_metrics = MLOpsMetrics()
        mlops_metrics.set_sys_reporting_status(True)

        # Notify MLOps with system information.
        while mlops_metrics.is_system_perf_reporting():
            mlops_metrics.report_system_metric()
            time.sleep(30)


if __name__ == "__main__":
    pass
