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
    Below are some default values for every endpoint. The
    `scale_loopback_interval_secs` parameter is used to specify
    the length of the interval between during which the autoscaler
    should check the next scaling operation.
    """
    min_replicas: int = 0
    max_replicas: int = 0
    current_replicas: int = 0
    scaleup_delay_secs: float = 300
    scaledown_delay_secs: float = 60
    endpoint_idleness_grace_period_secs: float = 300


class ReactivePolicy(AutoscalingPolicy):
    """
    Configuration parameters for the reactive autoscaling policy.
    EWM stands for Exponential Weighted Calculations, since we use
    the pandas.DataFrame.ewm() functionality.

    The following parameters refer to:
    - ewm_mins: the length of the interval we consider for reactive decision
    - ewm_alpha: the decay factor for the exponential weighted interval
    - ub_threshold: the upper bound scaling factor threshold for reactive decision
    - lb_threshold: the lower bound scaling factor threshold for reactive decision

    Example:

        Let's say that we consider 15 minutes as the length of our interval and a
        decay factor alpha with a value of 0.5:
            Original Sequence: [0.1, 0.2, 0.4, 3, 5, 10]
            EWM Sequence: [0.1, [0.166, 0.3, 1.74, 3.422, 6.763]

        If we assume that our previous scaling operation was triggered at value Y,
        then the conditions we use to decide whether to scale up or down are:
            Latency
                ScaleUP: X > ((1 + ub_threshold) * Y)
                ScaleDown: X < (lb_threshold * Y)
            QPS:
                ScaleUP: X < (lb_threshold * Y)
                ScaleDown: X < ((1 + ub_threshold) * Y)

        In other words, QPS is the inverse of Latency and vice versa.

    """
    metric: str = "latency"
    ewm_mins: int = 15
    ewm_alpha: float = 0.5
    ewm_latest: float = None
    ub_threshold: float = 0.5
    lb_threshold: float = 0.5
    triggering_value: float = None
    freeze_triggering_value: bool = False

    @field_validator("metric")
    def validate_option(cls, v):
        assert v in ["latency", "qps"]
        return v


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
                                   metrics: Dict) -> ScaleOp:

        # TODO(fedml-dimitris): TO BE COMPLETED!
        pass

    @classmethod
    def scale_operation_reactive(cls,
                                 reactive_policy: ReactivePolicy,
                                 metrics: Dict) -> ScaleOp:

        df = pd.DataFrame.from_records(metrics)
        df = df.set_index('timestamp')
        # timestamp is expected to be in micro-seconds, hence unit='us'.
        df.index = pd.to_datetime(df.index, unit="us")

        # Adding the context below to avoid having a series of warning messages.
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            short_period_data = df.last("{}min".format(reactive_policy.ewm_mins))
            # For instance, by using alpha = 0.1, we basically indicate that the most
            # recent values is weighted more. The reason is that the formula in pandas
            # is computed as:
            #   Yt = X_t + (1-a) * X_{t-1} + (1-a)^2 X_{t-2} / (1 + (1-a) + (1-a)^2)
            metric_name = "avg_latency" \
                if reactive_policy.metric == "latency" else "avg_qps"
            ewm_period = short_period_data[metric_name]\
                .ewm(alpha=reactive_policy.ewm_alpha).mean()

        latest_value = ewm_period.values[-1]
        # Just keep track / update the latest EWM value.
        reactive_policy.ewm_latest = latest_value
        # Assign the triggering value the first time we call the reactive
        # policy, if of course it has not been assigned already.
        if reactive_policy.triggering_value is None:
            reactive_policy.triggering_value = latest_value

        upper_bound = (1 + reactive_policy.ub_threshold) * reactive_policy.triggering_value
        lower_bound = reactive_policy.lb_threshold * reactive_policy.triggering_value

        scale_op = ScaleOp.NO_OP
        if latest_value <= lower_bound or latest_value >= upper_bound:
            # Replace the triggering value if the policy requests so.
            if not reactive_policy.freeze_triggering_value:
                reactive_policy.triggering_value = latest_value

            if reactive_policy.metric == "latency":
                # If the 'latency' is smaller than the
                # 'lower bound' then 'release' resources,
                # else if the 'latency' is 'greater' than
                # the 'upper bound' 'acquire' resources.
                if latest_value <= lower_bound:
                    scale_op = ScaleOp.DOWN_IN_OP
                elif latest_value >= upper_bound:
                    scale_op = ScaleOp.UP_OUT_OP
            elif reactive_policy.metric == "qps":
                # If the 'qps' is smaller than the
                # 'lower bound' then 'acquire' resources,
                # else if the 'qps' is 'greater' than
                # the 'upper bound' 'release' resources.
                if latest_value <= lower_bound:
                    scale_op = ScaleOp.UP_OUT_OP
                elif latest_value >= upper_bound:
                    scale_op = ScaleOp.DOWN_IN_OP

        return scale_op

    def run_autoscaling_policy(self,
                               autoscaling_policy: AutoscalingPolicy,
                               metrics: Dict) -> ScaleOp:

        if isinstance(autoscaling_policy, ReactivePolicy):
            scale_op = self.scale_operation_reactive(
                autoscaling_policy,
                metrics)
        elif isinstance(autoscaling_policy, PredictivePolicy):
            scale_op = self.scale_operation_predictive(
                autoscaling_policy,
                metrics)
        else:
            raise RuntimeError("Not a valid autoscaling policy instance.")

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

        if autoscaling_policy.current_replicas == 0:
            # if the current number_of requests > 1, then scale up/out
            scale_op = ScaleOp.NO_OP
        # if the current requests within the last X-minutes are 0
        # then scale down.
        else:
            # Fetch all previous timeseries values.
            endpoint_metrics = self.fedml_model_cache.get_endpoint_metrics(
                endpoint_id=endpoint_id)
            if not endpoint_metrics:
                # If no metrics are collected then do nothing.
                scale_op = ScaleOp.NO_OP
            else:
                # Trigger autoscaler with the metrics we have collected.
                scale_op = self.run_autoscaling_policy(autoscaling_policy, endpoint_metrics)
                # We cannot be lower than the minimum number of replicas,
                # nor exceed the maximum number of requested replicas.
                new_running_replicas = autoscaling_policy.current_replicas + scale_op.value
                if new_running_replicas <= autoscaling_policy.min_replicas:
                    scale_op = ScaleOp.NO_OP
                if new_running_replicas >= autoscaling_policy.max_replicas:
                    scale_op = ScaleOp.NO_OP

        return scale_op
