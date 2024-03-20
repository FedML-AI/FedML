import datetime
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import Autoscaler, ReactivePolicy
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.computing.scheduler.model_scheduler.autoscaler.test.traffic_simulation import TrafficSimulation
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache


def plot_qps_vs_latency_vs_scale(
        traffic,
        scale_operations,
        trend_lines=None):

    # plot
    fig, ax = plt.subplots(figsize=(30, 10))
    ts = [t[0] for t in traffic]
    ts = [datetime.datetime.strptime(t, TrafficSimulation.CONFIG_DATETIME_FORMAT) for t in ts]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    qps = [t[1] for t in traffic]
    latency = [t[2] for t in traffic]

    running_instances = [1]
    for scale_op in scale_operations[1:]:
        running_instances.append(running_instances[-1] + scale_op.value)

    # ax.plot_date(ts, qps, color='red', fmt="8--", linewidth=0.5, label="QPS")
    ax.plot_date(ts, latency, color='purple', fmt="p--", linewidth=0.5, label="Latency")
    ax.plot_date(ts, running_instances, color='green', fmt="*", linewidth=0.5, label="Instances")

    if trend_lines:
        for i, t in enumerate(trend_lines):
            ax.plot_date(ts, t, linestyle="solid", label="Trend Line {}".format(i))

    ax.set_xlabel("Timestamp")
    # ax.set_ylabel()
    # ax.set_yscale("log")
    plt.xticks(rotation=0)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax.grid(True)
    plt.legend()
    plot_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "plot/scaling_algorithm_test.png")

    plt.savefig(plot_file, bbox_inches='tight')


if __name__ == "__main__":

    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "tester", 0, 0, 0)
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
        latency_distribution="linear",
        num_values=300,
        submit_request_every_x_secs=30,
        reverse=False,
        with_warmup=False)

    # traffic = TrafficSimulation.generate_traffic_with_seasonality(
    #     num_values=1000,
    #     submit_request_every_x_secs=10,
    #     with_warmup=False)

    # Populate Redis and Trigger Autoscaler at every insertion
    autoscaler = Autoscaler(redis_addr, redis_port, redis_password)
    # autoscaling_policy = ReactivePolicy(metric="qps", lb_threshold=0, ub_threshold=3)
    autoscaling_policy = ReactivePolicy(metric="latency")
    scale_operations = []
    for i, t in enumerate(traffic):
        ts, qps, latency = t[0], t[1], t[2]
        fedml_model_cache.set_monitor_metrics(
            endpoint_id, "", "", "", latency, 0, 0, qps, 0, ts, 0)
        scale_op = autoscaler.scale_operation_endpoint(
            autoscaling_policy,
            str(endpoint_id))
        print(scale_op)
        scale_operations.append(scale_op)

    trend_lines = [
        # autoscaler.macd_vals,
        # autoscaler.macd_signal_line_vals,
        # autoscaler.ppo_vals,
        # autoscaler.ppo_cross_vals,
        # autoscaler.ppo_signal_line_vals
        # autoscaler.short_period_vals[-300:],
        # autoscaler.long_period_vals[-300:]
    ]
    plot_qps_vs_latency_vs_scale(
        traffic,
        scale_operations,
        trend_lines)
