import logging
import unittest
import time

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.policies import *
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import Autoscaler, ScaleOp
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog

ENV_REDIS_ADDR = "local"
ENV_REDIS_PORT = 6379
ENV_REDIS_PASSWD = "fedml_default"
ENV_ENDPOINT_ID_1 = 12345


class AutoscalerTest(unittest.TestCase):

    @classmethod
    def populate_redis_with_dummy_metrics(cls):
        fedml_model_cache = FedMLModelCache.get_instance()
        fedml_model_cache.set_redis_params(ENV_REDIS_ADDR, ENV_REDIS_PORT, ENV_REDIS_PASSWD)
        fedml_model_cache.set_monitor_metrics(
            ENV_ENDPOINT_ID_1, "", "", "", 5, 5, 5, 10, 100, 100, int(time.time_ns() / 1000), 0)

    @classmethod
    def clear_redis(cls):
        fedml_model_cache = FedMLModelCache.get_instance()
        # Clean up redis after test.
        fedml_model_cache.delete_endpoint_metrics(
            endpoint_ids=[ENV_ENDPOINT_ID_1])

    def test_autoscaler_singleton_pattern(self):
        autoscaler_1 = Autoscaler.get_instance()
        autoscaler_2 = Autoscaler.get_instance()
        # Only one object can be alive. Ensure both
        # autoscaler_{1,2} objects are the same.
        self.assertTrue(autoscaler_1 is autoscaler_2)

    def test_scale_operation_single_endpoint_ewm_policy(self):
        self.populate_redis_with_dummy_metrics()
        # Create autoscaler instance and define policy.
        autoscaler = Autoscaler.get_instance()
        latency_reactive_policy_default = {
            "current_replicas": 1,
            "min_replicas": 1,
            "max_replicas": 1,
            "metric": "ewm_latency",
            "ewm_mins": 15,
            "ewm_alpha": 0.5,
            "ub_threshold": 0.5,
            "lb_threshold": 0.5
        }
        autoscaling_policy = EWMPolicy(**latency_reactive_policy_default)
        scale_op_1 = autoscaler.scale_operation_endpoint(
            autoscaling_policy,
            endpoint_id=ENV_ENDPOINT_ID_1)

        # TODO Change to ScaleUP or ScaleDown not only not None.
        self.assertIsNotNone(scale_op_1)
        self.clear_redis()

    def test_scale_operation_single_endpoint_concurrency_query_policy(self):
        self.populate_redis_with_dummy_metrics()
        # Create autoscaler instance and define policy.
        autoscaler = Autoscaler.get_instance()
        concurrent_query_policy = {
            "current_replicas": 1,
            "min_replicas": 1,
            "max_replicas": 1,
            "queries_per_replica": 1,
            "window_size_secs": 60
        }
        autoscaling_policy = ConcurrentQueryPolicy(**concurrent_query_policy)
        scale_op_1 = autoscaler.scale_operation_endpoint(
            autoscaling_policy,
            endpoint_id=ENV_ENDPOINT_ID_1)

        # TODO Change to ScaleUP or ScaleDown not only not None.
        self.assertIsNotNone(scale_op_1)
        self.clear_redis()

    def test_scale_operation_single_endpoint_meet_traffic_demand_query_policy(self):
        self.populate_redis_with_dummy_metrics()
        # Create autoscaler instance and define policy.
        autoscaler = Autoscaler.get_instance()
        concurrent_query_policy = {
            "current_replicas": 1,
            "min_replicas": 1,
            "max_replicas": 1,
            "window_size_secs": 60
        }
        autoscaling_policy = MeetTrafficDemandPolicy(**concurrent_query_policy)
        scale_op_1 = autoscaler.scale_operation_endpoint(
            autoscaling_policy,
            endpoint_id=ENV_ENDPOINT_ID_1)

        # TODO Change to ScaleUP or ScaleDown not only not None.
        self.assertIsNotNone(scale_op_1)
        self.clear_redis()

    def test_validate_scaling_bounds(self):
        # Create autoscaler instance and define policy.
        autoscaler = Autoscaler.get_instance()
        autoscaling_policy = {
            "current_replicas": 2,
            "min_replicas": 1,
            "max_replicas": 3,
        }
        autoscaling_policy = AutoscalingPolicy(**autoscaling_policy)

        # Validate scale up.
        scale_up = autoscaler.validate_scaling_bounds(ScaleOp.UP_OUT_OP, autoscaling_policy)
        self.assertEqual(scale_up, ScaleOp.UP_OUT_OP)

        # Validate scale down.
        scale_down = autoscaler.validate_scaling_bounds(ScaleOp.DOWN_IN_OP, autoscaling_policy)
        self.assertEqual(scale_down, ScaleOp.DOWN_IN_OP)

        # Validate max out-of-bounds.
        autoscaling_policy.current_replicas = 3
        scale_oob_max = autoscaler.validate_scaling_bounds(ScaleOp.UP_OUT_OP, autoscaling_policy)
        self.assertEqual(scale_oob_max, ScaleOp.NO_OP)

        # Validate min out-of-bounds.
        autoscaling_policy.current_replicas = 1
        scale_oob_min = autoscaler.validate_scaling_bounds(ScaleOp.DOWN_IN_OP, autoscaling_policy)
        self.assertEqual(scale_oob_min, ScaleOp.NO_OP)

    def test_enforce_scaling_down_delay_interval(self):
        self.populate_redis_with_dummy_metrics()
        # Create autoscaler instance and define policy.
        autoscaler = Autoscaler.get_instance()
        autoscaling_policy = {
            "current_replicas": 1,
            "min_replicas": 1,
            "max_replicas": 1,
        }
        autoscaling_policy = AutoscalingPolicy(**autoscaling_policy)

        autoscaling_policy.scaledown_delay_secs = 0.0
        scale_down = autoscaler.enforce_scaling_down_delay_interval(ENV_ENDPOINT_ID_1, autoscaling_policy)
        self.assertEqual(scale_down, ScaleOp.DOWN_IN_OP)

        autoscaling_policy.scaledown_delay_secs = 1
        scale_noop = autoscaler.enforce_scaling_down_delay_interval(ENV_ENDPOINT_ID_1, autoscaling_policy)
        self.assertEqual(scale_noop, ScaleOp.NO_OP)

        time.sleep(2)
        scale_down = autoscaler.enforce_scaling_down_delay_interval(ENV_ENDPOINT_ID_1, autoscaling_policy)
        self.assertEqual(scale_down, ScaleOp.DOWN_IN_OP)
        self.clear_redis()


if __name__ == "__main__":
    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "server", 0, 0, 0)
    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.DEBUG)
    unittest.main()
