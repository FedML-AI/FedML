import math
import time
import warnings

import pandas as pd

from enum import Enum
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from pydantic import BaseModel, field_validator
from utils.singleton import Singleton
from typing import Dict


class ScaleOp(Enum):
    NO_OP = 0
    UP_OUT_OP = 1
    DOWN_IN_OP = -1


class AutoscalingPolicy(BaseModel):
    """
    Below are some default values for every endpoint.

    The following parameters refer to:
    - current_replicas: the number of currently running replicas of the endpoint
    - min_replicas: the minimum number of replicas of the endpoint in the instance group
    - max_replicas: the maximum number of replicas of the endpoint in the instance group
    - release_replica_after_idle_secs: when to release a single idle replica
    - scaledown_delay_secs: how many seconds to wait before performing a scale down operation
    - scaleup_cost_secs: how many seconds it takes/costs to perform a scale up operation
    - last_triggering_value: the last value that triggered a scaling operation

    The `replica_idle_grace_secs` parameter is used as
    the monitoring interval after which a running replica
    of an idle endpoint should be released.
    """
    current_replicas: int = 0
    min_replicas: int = 0
    max_replicas: int = 0
    release_replica_after_idle_secs: float = 300
    scaledown_delay_secs: float = 60
    scaleup_cost_secs: float = 300
    last_triggering_value: float = None


class EWMPolicy(AutoscalingPolicy):
    """
    Configuration parameters for the reactive autoscaling policy.
    EWM stands for Exponential Weighted Calculations, since we use
    the pandas.DataFrame.ewm() functionality.

    For panda's EWM using alpha = 0.1, we indicate that the most recent
    values are weighted more. The reason is that the exponential weighted
    mean formula in pandas is computed as:
        Yt = X_t + (1-a) * X_{t-1} + (1-a)^2 X_{t-2} / (1 + (1-a) + (1-a)^2)

    The following parameters refer to:
    - ewm_mins: the length of the interval we consider for reactive decision
    - ewm_alpha: the decay factor for the exponential weighted interval
    - ewm_latest: the latest recorded value of the metric's exponential weighted mean
    - ub_threshold: the upper bound scaling factor threshold for reactive decision
    - lb_threshold: the lower bound scaling factor threshold for reactive decision

    Example:

        Let's say that we consider 15 minutes as the length of our interval and a
        decay factor alpha with a value of 0.5:
            Original Sequence: [0.1, 0.2, 0.4, 3, 5, 10]
            EWM Sequence: [0.1, [0.166, 0.3, 1.74, 3.422, 6.763]

        If we assume that our previous scaling operation was triggered at value Y,
        then the conditions we use to decide whether to scale up or down are:
            Latency:
                ScaleUP: X > ((1 + ub_threshold) * Y)
                ScaleDown: X < (lb_threshold * Y)
            QPS:
                ScaleUP: X < (lb_threshold * Y)
                ScaleDown: X < ((1 + ub_threshold) * Y)

        In other words, QPS is the inverse of Latency and vice versa.
    """
    metric: str = "ewm_latency"
    ewm_mins: int = 15
    ewm_alpha: float = 0.5
    ewm_latest: float = None
    ub_threshold: float = 0.5
    lb_threshold: float = 0.5

    @field_validator("metric")
    def validate_option(cls, v):
        assert v in ["ewm_latency", "ewm_qps"]
        return v


class ConcurrentQueryPolicy(AutoscalingPolicy):
    """
    This policy captures the number of queries we want to support
    per replica over the defined window length in seconds.
    """
    queries_per_replica: int = 1
    window_size_secs: int = 60


class PredictivePolicy(AutoscalingPolicy):
    # TODO(fedml-dimitris): TO BE COMPLETED!
    pass


class Autoscaler(metaclass=Singleton):

    def __init__(self, redis_addr="local", redis_port=6379, redis_password="fedml_default"):
        super().__init__()
        self.fedml_model_cache = FedMLModelCache.get_instance()
        self.fedml_model_cache.set_redis_params(redis_addr, redis_port, redis_password)

    @staticmethod
    def get_instance(*args, **kwargs):
        return Autoscaler(*args, **kwargs)

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
            short_period_data = metrics.last("{}min".format(ewm_policy.ewm_mins))
            metric_name = "current_latency" \
                if "ewm_latency" == ewm_policy.metric else "current_qps"
            ewm_period = short_period_data[metric_name] \
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
        if ewm_policy.last_triggering_value is None:
            ewm_policy.last_triggering_value = latest_value

        upper_bound = (1 + ewm_policy.ub_threshold) * ewm_policy.last_triggering_value
        lower_bound = ewm_policy.lb_threshold * ewm_policy.last_triggering_value

        if latest_value <= lower_bound or latest_value >= upper_bound:
            # Replace the triggering value if the policy requests so.
            ewm_policy.last_triggering_value = latest_value

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
            queries_num = period_data.shape[0]

        # QSR: Queries per Second per Replica: (Number of Queries / Number of Current Replicas) / Window Size
        # Comparing target QSR to current QSR.
        target_qsr = \
            concurrent_query_policy.queries_per_replica / concurrent_query_policy.window_size_secs
        # We need to floor the target queries per replica, therefore we need to ceil the division
        # to ensure we will not have too much fluctuation. For instance, if the user requested to
        # support 2 queries per replica per 60 seconds, the target QSR is 2/60 = 0.0333.
        # Then, if we had 5 queries sent to 3 replicas in 60 seconds, the current QSR
        # would be (5/3)/60 = 0.0277. To avoid the fluctuation, we need to round the incoming
        # number of queries per replica to the nearest integer and then divide by the window size.
        current_qsr = \
            (math.ceil(queries_num / concurrent_query_policy.current_replicas) /
             concurrent_query_policy.window_size_secs)

        if current_qsr > target_qsr:
            concurrent_query_policy.last_triggering_value = current_qsr
            scale_op = ScaleOp.UP_OUT_OP
        elif current_qsr < target_qsr:
            concurrent_query_policy.last_triggering_value = current_qsr
            scale_op = ScaleOp.DOWN_IN_OP
        else:
            scale_op = ScaleOp.NO_OP

        return scale_op

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
        most_recent_metric = self.fedml_model_cache.get_endpoint_metrics(
            endpoint_id=endpoint_id,
            k_recent=1)

        # Default to nothing.
        scale_op = ScaleOp.NO_OP
        if not most_recent_metric:
            # If no metric exists then no scaling operation.
            return scale_op

        # If we continue here, then it means that there was at least one request.
        # The `most_recent_metric` is of type list, hence we need to access index 0.
        most_recent_metric = most_recent_metric[0]
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
                # Else, trigger the autoscaling policy. Fetch all previous
                # timeseries values. We do not check if the list is empty,
                # since we already have past requests.
                metrics = self.fedml_model_cache.get_endpoint_metrics(
                    endpoint_id=endpoint_id)
                metrics_df = pd.DataFrame.from_records(metrics)
                metrics_df = metrics_df.set_index('timestamp')
                # timestamp is expected to be in micro-seconds, hence unit='us'.
                metrics_df.index = pd.to_datetime(metrics_df.index, unit="us")

                # Trigger autoscaler with the metrics we have collected.
                scale_op = self.run_autoscaling_policy(
                    autoscaling_policy=autoscaling_policy,
                    metrics=metrics_df)

        # Finally, check the scaling bounds of the endpoint
        # before triggering the scaling operation.
        scale_op = self.validate_scaling_bounds(
            scale_op=scale_op,
            autoscaling_policy=autoscaling_policy)

        return scale_op
