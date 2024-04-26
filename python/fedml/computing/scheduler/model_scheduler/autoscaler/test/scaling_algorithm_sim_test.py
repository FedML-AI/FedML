import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import \
    Autoscaler, EWMPolicy, ConcurrentQueryPolicy, MeetTrafficDemandPolicy
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.computing.scheduler.model_scheduler.autoscaler.test.traffic_simulation import TrafficSimulation
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache


def plot_qps_vs_latency_vs_scale(
        metric,
        traffic,
        scale_operations,
        trend_lines=None,
        triggering_points=None):

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

    if "qps" in metric:
        ax.plot_date(ts, qps, color='purple', fmt="8--", linewidth=0.5, label="QPS")
    elif "latency" in metric:
        ax.plot_date(ts, latency, color='purple', fmt="p--", linewidth=0.5, label="Latency")
    ax.plot_date(ts, running_instances, color='green', fmt="*", linewidth=0.5, label="Instances")

    if trend_lines:
        for i, t in enumerate(trend_lines):
            ax.plot_date(ts, t, linestyle="solid", label="Trend Line {}".format(i))

    if triggering_points:
        ax.plot_date(ts, triggering_points, fmt="", marker="^", markersize=12, label="Triggering Points", color="red")

    ax.set_xlabel("Timestamp")
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--redis_addr', default="local")
    parser.add_argument('--redis_port', default=6379)
    parser.add_argument('--redis_password', default="fedml_default")
    parser.add_argument('--endpoint_id', default=12345)
    parser.add_argument('--metric',
                        default="query_concurrency",
                        help="Either ewm_latency, ewm_qps, query_concurrency, meet_traffic_demand")
    parser.add_argument('--distribution',
                        default="seasonal",
                        help="Either random, linear, exponential or seasonal.")
    args = parser.parse_args()

    fedml_model_cache = FedMLModelCache.get_instance()
    fedml_model_cache.set_redis_params(args.redis_addr, args.redis_port, args.redis_password)

    # INFO To test different distributions, simply change the distribution value
    # to the following possible values:
    #   "random", "linear", "exponential"
    # Moreover, you can also play around with the order of values in an
    # ascending (reverse=False) or descending (reverse=True) order.
    start_date = datetime.datetime.strptime(
        datetime.datetime.now().strftime(TrafficSimulation.CONFIG_DATETIME_FORMAT),
        TrafficSimulation.CONFIG_DATETIME_FORMAT)
    if args.distribution in ["random", "linear", "exponential"]:
        traffic_dist = TrafficSimulation(start_date=start_date).generate_traffic(
            qps_distribution=args.distribution,
            latency_distribution=args.distribution,
            num_values=300,
            submit_request_every_x_secs=1,
            reverse=False,
            with_warmup=False)
    elif args.distribution == "seasonal":
        traffic_dist = TrafficSimulation(start_date=start_date).generate_traffic_with_seasonality(
            num_values=1000,
            submit_request_every_x_secs=30,
            with_warmup=False)
    else:
        raise RuntimeError("Not a supported distribution")

    # INFO Please remember to change these two variables below when attempting
    # to test the simulation of the autoscaling policy simulation.
    testing_metric = args.metric
    policy_config = dict()
    policy_config["min_replicas"] = 1  # Always 1.
    policy_config["max_replicas"] = 1000  # Unlimited.
    policy_config["current_replicas"] = 1
    policy_config["scaledown_delay_secs"] = 0

    if testing_metric == "ewm_latency":
        policy_config.update({
            "metric": "ewm_latency", "ewm_mins": 15, "ewm_alpha": 0.5, "ub_threshold": 0.5, "lb_threshold": 0.5
        })
        autoscaling_policy = EWMPolicy(**policy_config)
    elif testing_metric == "ewm_qps":
        policy_config.update({
            "metric": "ewm_qps", "ewm_mins": 15, "ewm_alpha": 0.5, "ub_threshold": 2, "lb_threshold": 0.5
        })
        autoscaling_policy = EWMPolicy(**policy_config)
    elif testing_metric == "query_concurrency":
        policy_config.update({
            "queries_per_replica": 2, "window_size_secs": 60
        })
        autoscaling_policy = ConcurrentQueryPolicy(**policy_config)
    elif testing_metric == "meet_traffic_demand":
        policy_config.update({
            "window_size_secs": 60
        })
        autoscaling_policy = MeetTrafficDemandPolicy(**policy_config)
    else:
        raise RuntimeError("Please define a valid policy metric.")

    print(policy_config)
    autoscaler = Autoscaler.get_instance(args.redis_addr, args.redis_port, args.redis_password)

    scale_operations = []
    ewm_values = []
    triggering_values = []
    for i, t in enumerate(traffic_dist):
        ts, qps, latency = t[0], t[1], t[2]
        # We convert the timestamp to epoch time with microseconds, since this
        # is the expected input in the REDIS database for the timestamp column.
        ts_epoch = int(datetime.datetime.strptime(
            ts, TrafficSimulation.CONFIG_DATETIME_FORMAT).timestamp() * 1e6)
        fedml_model_cache.set_monitor_metrics(
            end_point_id=args.endpoint_id,
            end_point_name="",
            model_name="",
            model_version="",
            total_latency=latency,
            avg_latency=latency,
            current_latency=latency,
            total_request_num=i,
            current_qps=qps,
            avg_qps=qps,
            timestamp=ts_epoch,
            device_id=0)
        scale_op = autoscaler.scale_operation_endpoint(
            autoscaling_policy,
            str(args.endpoint_id))
        autoscaling_policy.current_replicas = \
            autoscaling_policy.current_replicas + scale_op.value
        if isinstance(autoscaling_policy, EWMPolicy):
            ewm_values.append(autoscaling_policy.ewm_latest)
        triggering_values.append(autoscaling_policy.previous_triggering_value)
        scale_operations.append(scale_op)

    triggering_values_to_plot = []
    for idx, v in enumerate(triggering_values):
        if (idx - 1) < 0 or triggering_values[idx] == triggering_values[idx - 1]:
            triggering_values_to_plot.append(None)
        else:
            triggering_values_to_plot.append(v)

    trend_lines = [ewm_values] if ewm_values else None
    plot_qps_vs_latency_vs_scale(
        metric=testing_metric,
        traffic=traffic_dist,
        scale_operations=scale_operations,
        trend_lines=trend_lines,
        triggering_points=triggering_values_to_plot)

    # Clear redis monitor keys.
    fedml_model_cache.delete_endpoint_metrics(
        endpoint_ids=[args.endpoint_id])
