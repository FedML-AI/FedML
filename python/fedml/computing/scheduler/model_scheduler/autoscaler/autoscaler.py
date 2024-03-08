import conf
import fedml
import logging
import os

import datetime as dt
import pandas as pd

from utils.singleton import Singleton

from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache


class FedMLAutoscaler(metaclass=Singleton):

    def __init__(self, redis_addr="local", redis_port=6379, redis_password="fedml_default"):
        super().__init__()
        self.fedml_model_cache = FedMLModelCache.get_instance()
        self.fedml_model_cache.set_redis_params(redis_addr, redis_port, redis_password)

    @staticmethod
    def get_instance():
        return FedMLAutoscaler()

    def scale_operation(self, metrics):
        print(metrics)
        timestamp_to_seconds = lambda x: x / 1e6
        convert_timestamp = \
            lambda x: dt.datetime.fromtimestamp(x).strftime(conf.CONFIG_DATETIME_FORMAT)
        metrics = [(m["current_qps"], convert_timestamp(timestamp_to_seconds(m["timestamp"]))) for m in metrics]
        print(metrics)
        # (-1, 0, 1)
        return 0

    def scale_operation_single_endpoint(self,
                                        endpoint_id,
                                        timeseries_length=None):

        if not timeseries_length:
            timeseries_length = conf.CONFIG_TIMESERIES_LENGTH

        endpoint_metrics = self.fedml_model_cache.get_endpoint_metrics(
            endpoint_id=endpoint_id,
            k_recent=timeseries_length)

        scale_op = self.scale_operation(endpoint_metrics)
        return scale_op

    def scale_operation_all_endpoints(self, timeseries_length=None):
        scale_operation = dict()
        endpoint_ids = self.fedml_model_cache.get_endpoints_ids()
        for eid in endpoint_ids:
            scale_op = self.scale_operation_single_endpoint(
                endpoint_id=eid,
                timeseries_length=timeseries_length)
            scale_operation[eid] = scale_op
