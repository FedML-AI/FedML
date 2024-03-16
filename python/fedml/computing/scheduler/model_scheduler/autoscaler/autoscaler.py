import fedml
import logging

import os

import datetime as dt
import fedml.computing.scheduler.model_scheduler.autoscaler.conf as conf
import pandas as pd

from dataclasses import dataclass
from enum import Enum
from utils.singleton import Singleton
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache


class ScaleOp(Enum):
    NO_OP = 0
    UP_OUT_OP = 1
    DOWN_IN_OP = 2


@dataclass
class AutoscalingPolicy(object):
    """
    Below are some default values for every endpoint. The
    `scale_loopback_interval_secs` parameter is used to specify
    the length of the interval between during which the autoscaler
    should check the next scaling operation.
    """
    min_replicas: int = 1
    max_replicas: int = 1
    running_replicas: int = 0
    scale_loopback_interval_secs: float = 60
    scaleup_delay_secs: float = 300
    scaledown_delay_secs: float = 60
    endpoint_idleness_grace_period_secs: float = 300


class Autoscaler(metaclass=Singleton):

    def __init__(self, redis_addr="local", redis_port=6379, redis_password="fedml_default"):
        super().__init__()
        self.fedml_model_cache = FedMLModelCache.get_instance()
        self.fedml_model_cache.set_redis_params(redis_addr, redis_port, redis_password)

    @staticmethod
    def get_instance():
        return Autoscaler()

    def scale_operation(self, metrics):
        timestamp_to_seconds = lambda x: x / 1e6
        convert_timestamp = \
            lambda x: dt.datetime.fromtimestamp(x).strftime(conf.CONFIG_DATETIME_FORMAT)
        metrics = [(m["current_qps"], convert_timestamp(timestamp_to_seconds(m["timestamp"]))) for m in metrics]
        print(metrics)
        # (-1, 0, 1)
        return ScaleOp.NO_OP

    def scale_operation_single_endpoint(self,
                                        autoscaling_policy: AutoscalingPolicy,
                                        endpoint_id: str,
                                        timeseries_length=None) -> ScaleOp:
        """
        Decision rules:
            (1) if min == 0 then decide if we need to increase replicas (scale up/out).
            (2) if min == max then decide if we need to reduce replicas (scale down/in).
            (3) if min < max then decide if we need to
                - increase (scale up/out) or
                - reduce replicas (scale down/in)
            (4) By default, we do nothing.

        Return:
            +1 : increase replicas by 1
            -1 : decrease replicas by 1
            0: do nothing
        """

        scale_op = 0

        # TODO Get minimum and maximum replicas from Redis
        min_replicas = 0
        max_replicas = 10

        # if no requests the last 5 minutes scale down.


        if autoscaling_policy.min_replicas == 0:
            # if number_of requests > 1, then scale up/out
            pass
        elif autoscaling_policy.min_replicas == max_replicas:
            # if latency is relatively
            pass
        else:
            if not timeseries_length:
                timeseries_length = conf.CONFIG_TIMESERIES_LENGTH
            endpoint_metrics = self.fedml_model_cache.get_endpoint_metrics(
                endpoint_id=endpoint_id,
                k_recent=timeseries_length)
            scale_op = self.scale_operation(endpoint_metrics)
        logging.info("Endpoint: {}, Scaling operation decision: {}".format(
            endpoint_id, scale_op))
        return scale_op

    def scale_operation_all_endpoints(self, timeseries_length=None):
        scale_operation = dict()
        endpoint_ids = self.fedml_model_cache.get_endpoints_ids()
        for eid in endpoint_ids:
            scale_op = self.scale_operation_single_endpoint(
                endpoint_id=eid,
                timeseries_length=timeseries_length)
            scale_operation[eid] = scale_op
