import logging
import math
import time
import warnings

import pandas as pd

from enum import Enum
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from policies import *
from utils.singleton import Singleton


class ScaleOp(Enum):
    NO_OP = 0
    UP_OUT_OP = 1
    DOWN_IN_OP = -1


class Autoscaler(metaclass=Singleton):

    def __init__(self, redis_addr="local", redis_port=6379, redis_password="fedml_default"):
        super().__init__()
        self.fedml_model_cache = FedMLModelCache.get_instance()
        self.fedml_model_cache.set_redis_params(redis_addr, redis_port, redis_password)

    @staticmethod
    def get_instance(*args, **kwargs):
        return Autoscaler(*args, **kwargs)

    @classmethod
    def get_current_timestamp_micro_seconds(cls):
        # in REDIS we record/operate in micro-seconds, hence the division by 1e3!
        return int(format(time.time_ns() / 1000.0, '.0f'))

    @classmethod
    def scale_operation_predictive(cls,
                                   predictive_policy: PredictivePolicy,
                                   metrics: pd.DataFrame) -> ScaleOp:

        # TODO(fedml-dimitris): TO BE COMPLETED!
        pass

    @classmethod
    def scale_operation_ewm(cls,
                            ewm_policy: EWMPolicy,
                            metrics: pd.DataFrame) -> ScaleOp:

        # Adding the context below to avoid having a series of warning messages.
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            period_data = metrics.last("{}min".format(ewm_policy.ewm_mins))
            # If the data frame window is empty then do nothing more, just return.
            if period_data.empty:
                return ScaleOp.NO_OP
            metric_name = "current_latency" \
                if "ewm_latency" == ewm_policy.metric else "current_qps"
            ewm_period = period_data[metric_name] \
                .ewm(alpha=ewm_policy.ewm_alpha).mean()

        scale_op = ScaleOp.NO_OP
        # If there is no exponential moving average within this
        # time frame, then no scaling operation takes place.
        if len(ewm_period.values) == 0:
            return scale_op

        latest_value = ewm_period.values[-1]
        # Just keep track / update the latest EWM value.
        ewm_policy.ewm_latest = latest_value
        # Assign the triggering value the first time we call the reactive
        # policy, if of course it has not been assigned already.
        if ewm_policy.previous_triggering_value is None:
            ewm_policy.previous_triggering_value = latest_value

        upper_bound = (1 + ewm_policy.ub_threshold) * ewm_policy.previous_triggering_value
        lower_bound = ewm_policy.lb_threshold * ewm_policy.previous_triggering_value

        if latest_value <= lower_bound or latest_value >= upper_bound:
            # Replace the triggering value if the policy requests so.
            ewm_policy.previous_triggering_value = latest_value

            if ewm_policy.metric == "ewm_latency":
                # If the 'latency' is smaller than the
                # 'lower bound' then 'release' resources,
                # else if the 'latency' is 'greater' than
                # the 'upper bound' 'acquire' resources.
                if latest_value <= lower_bound:
                    scale_op = ScaleOp.DOWN_IN_OP
                elif latest_value >= upper_bound:
                    scale_op = ScaleOp.UP_OUT_OP
            elif ewm_policy.metric == "ewm_qps":
                # If the 'qps' is smaller than the
                # 'lower bound' then 'acquire' resources,
                # else if the 'qps' is 'greater' than
                # the 'upper bound' 'release' resources.
                if latest_value <= lower_bound:
                    scale_op = ScaleOp.UP_OUT_OP
                elif latest_value >= upper_bound:
                    scale_op = ScaleOp.DOWN_IN_OP

        return scale_op

    @classmethod
    def scale_operation_query_concurrency(cls,
                                          concurrent_query_policy: ConcurrentQueryPolicy,
                                          metrics: pd.DataFrame) -> ScaleOp:

        # Adding the context below to avoid having a series of warning messages.
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            # Here, the number of queries is the number of rows in the short period data frame.
            period_data = metrics.last("{}s".format(concurrent_query_policy.window_size_secs))
            # If the data frame window is empty then do nothing more, just return.
            if period_data.empty:
                return ScaleOp.NO_OP
            queries_num = period_data.shape[0]

        try:
            # QSR: Queries per Second per Replica: (Number of Queries / Number of Current Replicas) / Window Size
            # Comparing target QSR to current QSR.
            target_qrs = \
                concurrent_query_policy.queries_per_replica / concurrent_query_policy.window_size_secs
            # We need to floor the target queries per replica, therefore we need to ceil the division
            # to ensure we will not have too much fluctuation. For instance, if the user requested to
            # support 2 queries per replica per 60 seconds, the target QSR is 2/60 = 0.0333.
            # Then, if we had 5 queries sent to 3 replicas in 60 seconds, the current QSR
            # would be (5/3)/60 = 0.0277. To avoid the fluctuation, we need to round the incoming
            # number of queries per replica to the nearest integer and then divide by the window size.
            current_qrs = \
                (math.ceil(queries_num / concurrent_query_policy.current_replicas) /
                 concurrent_query_policy.window_size_secs)
        except ZeroDivisionError as error:
            logging.error("Division by zero.")
            return ScaleOp.NO_OP

        if current_qrs > target_qrs:
            concurrent_query_policy.previous_triggering_value = current_qrs
            scale_op = ScaleOp.UP_OUT_OP
        elif current_qrs < target_qrs:
            concurrent_query_policy.previous_triggering_value = current_qrs
            scale_op = ScaleOp.DOWN_IN_OP
        else:
            scale_op = ScaleOp.NO_OP

        return scale_op

    @classmethod
    def scale_operation_meet_traffic_demand(cls,
                                            meet_traffic_demand_policy: MeetTrafficDemandPolicy,
                                            metrics: pd.DataFrame) -> ScaleOp:

        # Adding the context below to avoid having a series of warning messages.
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            # Here, the number of queries is the number of rows in the short period data frame.
            period_data = metrics.last("{}s".format(meet_traffic_demand_policy.window_size_secs))
            # If the data frame window is empty then do nothing more, just return.
            if period_data.empty:
                return ScaleOp.NO_OP

        period_requests_num = period_data.shape[0]
        all_latencies = metrics["current_latency"]
        # Original value is milliseconds, convert to seconds.
        average_latency = all_latencies.mean() / 1e3

        try:
            # RS: Requests_per_Second
            rs = period_requests_num / meet_traffic_demand_policy.window_size_secs
            # QS: Queries_per_Second
            qs = 1 / average_latency
        except ZeroDivisionError as error:
            logging.error("Division by zero.")
            return ScaleOp.NO_OP

        if rs > qs:
            # Need to meet the demand.
            return ScaleOp.UP_OUT_OP
        elif rs < qs:
            # Demand already met.
            return ScaleOp.DOWN_IN_OP
        else:
            return ScaleOp.NO_OP

    def run_autoscaling_policy(self,
                               autoscaling_policy: AutoscalingPolicy,
                               metrics: pd.DataFrame) -> ScaleOp:

        if isinstance(autoscaling_policy, EWMPolicy):
            scale_op = self.scale_operation_ewm(
                autoscaling_policy,
                metrics)
        elif isinstance(autoscaling_policy, ConcurrentQueryPolicy):
            scale_op = self.scale_operation_query_concurrency(
                autoscaling_policy,
                metrics)
        elif isinstance(autoscaling_policy, MeetTrafficDemandPolicy):
            scale_op = self.scale_operation_meet_traffic_demand(
                autoscaling_policy,
                metrics)
        elif isinstance(autoscaling_policy, PredictivePolicy):
            scale_op = self.scale_operation_predictive(
                autoscaling_policy,
                metrics)
        else:
            raise RuntimeError("Not a valid autoscaling policy instance.")

        return scale_op

    @classmethod
    def validate_scaling_bounds(cls,
                                scale_op: ScaleOp,
                                autoscaling_policy: AutoscalingPolicy) -> ScaleOp:
        # We cannot be lower than the minimum number of replicas,
        # nor exceed the maximum number of requested replicas.
        new_running_replicas = autoscaling_policy.current_replicas + scale_op.value
        if new_running_replicas < autoscaling_policy.min_replicas:
            scale_op = ScaleOp.NO_OP
        elif new_running_replicas > autoscaling_policy.max_replicas:
            scale_op = ScaleOp.NO_OP
        return scale_op

    def enforce_scaling_down_delay_interval(self,
                                            endpoint_id,
                                            autoscaling_policy: AutoscalingPolicy) -> ScaleOp:
        """
        This function checks if scaling down delay seconds set by the policy
        has been exceeded. To enforce the delay it uses REDIS to persist the
        time of the scaling down operation.

        If such a record exists it fetches the previous scale down operation's timestamp
        and compares the duration of the interval (delay).

        If the interval is exceeded then it triggers/allows the scaling operation to be
        passed to the calling process, else it returns a no operation.
        """

        # Get the timestamp of the previous scaling down timestamp (if any).
        previous_timestamp = \
            self.fedml_model_cache.get_endpoint_scaling_down_decision_time(endpoint_id)
        current_timestamp = self.get_current_timestamp_micro_seconds()
        diff_secs = (current_timestamp - previous_timestamp) / 1e6
        if diff_secs >= autoscaling_policy.scaledown_delay_secs:
            # At this point, we will perform the scaling down operation, hence
            # we need to delete the previously stored scaling down timestamp (if any).
            self.fedml_model_cache.delete_endpoint_scaling_down_decision_time(endpoint_id)
            scale_op = ScaleOp.DOWN_IN_OP
        else:
            # Record the timestamp of the scaling down operation.
            self.fedml_model_cache.set_endpoint_scaling_down_decision_time(
                endpoint_id, current_timestamp)
            scale_op = ScaleOp.NO_OP

        return scale_op

    def scale_operation_endpoint(self,
                                 autoscaling_policy: AutoscalingPolicy,
                                 endpoint_id: str) -> ScaleOp:
        """
        Decision rules:
            (1) if current_replicas == 0 then decide if we need to increase (scale up/out).
            (2) if current_replicas <= max then decide if we need to
                - increase replicas (scale up/out) or
                - reduce replicas (scale down/in)
            (3) By default, we do nothing.

        Return:
            +1 : increase replicas by 1
            -1 : decrease replicas by 1
            0: do nothing
        """

        # Fetch most recent metric record from the database.
        metrics = self.fedml_model_cache.get_endpoint_metrics(
            endpoint_id=endpoint_id)

        # Default to nothing.
        scale_op = ScaleOp.NO_OP
        if not metrics:
            # If no metric exists then no scaling operation.
            return scale_op

        # If we continue here, then it means that there was at least one request.
        # The `most_recent_metric` is of type list, hence we need to access index 0.
        most_recent_metric = metrics[-1]
        latest_request_timestamp_micro_secs = most_recent_metric["timestamp"]
        # The time module does not have a micro-second function built-in, so we need to
        # divide nanoseconds by 1e3 and convert to micro-seconds.
        current_time_micro_seconds = time.time_ns() / 1e3
        # compute elapsed time and convert to seconds
        elapsed_time_secs = \
            (current_time_micro_seconds - latest_request_timestamp_micro_secs) / 1e6
        if elapsed_time_secs > autoscaling_policy.release_replica_after_idle_secs:
            # If the elapsed time is greater than the requested idle time,
            # in other words there was no incoming request then scale down.
            scale_op = ScaleOp.DOWN_IN_OP
        else:
            # Otherwise, it means there was a request within the elapsed time, then:
            if autoscaling_policy.current_replicas == 0:
                # Check if the current number of running replicas is 0,
                # then we need more resources, hence ScaleOp.UP_OUT_OP.
                scale_op = ScaleOp.UP_OUT_OP
            else:
                # Else, trigger the autoscaling policy with all existing values.
                metrics_df = pd.DataFrame.from_records(metrics)
                metrics_df = metrics_df.set_index('timestamp')
                # timestamp is expected to be in micro-seconds, hence unit='us'.
                metrics_df.index = pd.to_datetime(metrics_df.index, unit="us")
                # Decide scaling operation given all metrics.
                scale_op = self.run_autoscaling_policy(
                    autoscaling_policy=autoscaling_policy,
                    metrics=metrics_df)

        # Check the scaling bounds of the endpoint
        # before triggering the scaling operation.
        scale_op = self.validate_scaling_bounds(
            scale_op=scale_op,
            autoscaling_policy=autoscaling_policy)

        # If the scaling decision is a scale down operation, then perform
        # a final check to ensure the scaling down grace period is satisfied.
        if scale_op == scale_op.DOWN_IN_OP:
            scale_op = self.enforce_scaling_down_delay_interval(
                endpoint_id, autoscaling_policy)

        return scale_op
