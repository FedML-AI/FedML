import logging
import unittest
import time

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import Autoscaler, ReactivePolicy
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog

ENV_REDIS_ADDR = "local"
ENV_REDIS_PORT = 6379
ENV_REDIS_PASSWD = "fedml_default"
ENV_ENDPOINT_ID_1 = 12345
ENV_ENDPOINT_ID_2 = 77777


class AutoscalerTest(unittest.TestCase):

    def test_autoscaler_singleton_pattern(self):
        autoscaler_1 = Autoscaler.get_instance()
        autoscaler_2 = Autoscaler.get_instance()
        # Only one object can be alive. Ensure both
        # autoscaler_{1,2} objects are the same.
        self.assertTrue(autoscaler_1 is autoscaler_2)

    def test_scale_operation_single_endpoint_reactive(self):

        # Populate redis with some dummy values for each endpoint before running the test.
        fedml_model_cache = FedMLModelCache.get_instance()
        fedml_model_cache.set_redis_params(ENV_REDIS_ADDR, ENV_REDIS_PORT, ENV_REDIS_PASSWD)
        fedml_model_cache.set_monitor_metrics(
            ENV_ENDPOINT_ID_1, "", "", "", 5, 5, 10, 100, 100, int(time.time_ns() / 1000), 0)
        fedml_model_cache.set_monitor_metrics(
            ENV_ENDPOINT_ID_1, "", "", "", 5, 5, 10, 100, 100, int(time.time_ns() / 1000), 0)

        # Create autoscaler instance and define policy.
        autoscaler = Autoscaler.get_instance()
        latency_reactive_policy_default = {
            "min_replicas": 1,
            "max_replicas": 1,
            "current_replicas": 1,
            "metric": "latency",
            "ewm_mins": 15,
            "ewm_alpha": 0.5,
            "ub_threshold": 0.5,
            "lb_threshold": 0.5
        }

        autoscaling_policy = ReactivePolicy(**latency_reactive_policy_default)
        scale_op_1 = autoscaler.scale_operation_endpoint(
            autoscaling_policy,
            endpoint_id=ENV_ENDPOINT_ID_1)
        scale_op_2 = autoscaler.scale_operation_endpoint(
            autoscaling_policy,
            endpoint_id=ENV_ENDPOINT_ID_2)

        # Clean up redis after test.
        fedml_model_cache.delete_model_endpoint_metrics(
            endpoint_ids=[ENV_ENDPOINT_ID_1, ENV_ENDPOINT_ID_2])

        self.assertIsNotNone(scale_op_1)
        self.assertIsNotNone(scale_op_2)


if __name__ == "__main__":
    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "server", 0, 0, 0)
    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.DEBUG)
    unittest.main()
