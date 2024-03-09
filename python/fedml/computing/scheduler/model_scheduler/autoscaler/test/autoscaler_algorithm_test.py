import datetime
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import FedMLAutoscaler
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.computing.scheduler.model_scheduler.autoscaler.test.traffic_simulation import TrafficSimulation
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache


def plot_qps_vs_latency_vs_scale(traffic, scale_operations):

    # plot
    fig, ax = plt.subplots(figsize=(30, 10))
    ts = [t[0] for t in traffic]
    ts = [datetime.datetime.strptime(t, TrafficSimulation.CONFIG_DATETIME_FORMAT) for t in ts]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    qps = [t[1] for t in traffic]
    latency = [t[2] for t in traffic]

    running_instances = []
    for res in scale_operations:
        scale_op = 0
        if res == 1:
            scale_op = 1
        if res == -1:
            scale_op = -1

        # If it is the first time we populate the list then,
        # assign the default scale operation, else use
        # augment according to scale_op value: -1, 0, +1
        if not running_instances:
            running_instances.append(scale_op)
        else:
            running_instances.append(running_instances[-1] + scale_op)

    ax.plot_date(ts, qps, color='red', fmt="8--", linewidth=0.5, label="QPS")
    ax.plot_date(ts, latency, color='purple', fmt="p--", linewidth=0.5, label="Latency")
    ax.plot_date(ts, running_instances, color='green', fmt="*", linewidth=0.5, label="Instances")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("QPS")
    plt.xticks(rotation=0)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.grid(True)
    plt.legend()
    plot_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "plot/autoscaler_algorithm_scaling_ops.png")

    plt.savefig(plot_file, bbox_inches='tight')


if __name__ == "__main__":

    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "server", 0, 0, 0)
    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.DEBUG)

    redis_addr = "local"
    redis_port = 6379
    redis_password = "fedml_default"
    endpoint_id = 12345
    fedml_model_cache = FedMLModelCache.get_instance()
    fedml_model_cache.set_redis_params(redis_addr, redis_port, redis_password)
    fedml_model_cache.delete_model_endpoint_metrics(
        endpoint_id=endpoint_id)

    traffic = TrafficSimulation.generate_traffic(
        qps_distribution="random",
        latency_distribution="random",
        num_values=300,
        submit_request_every_x_secs=30,
        with_warmup=True)

    # Populate Redis
    for t in traffic:
        ts, qps, latency = t[0], t[1], t[2]
        fedml_model_cache.set_monitor_metrics(
            endpoint_id, "", "", "", latency, 0, 0, qps, 0, ts, 0)

    # TODO Trigger Autoscaler.
    scale_operations = [0 for _ in traffic]
    plot_qps_vs_latency_vs_scale(traffic, scale_operations)
