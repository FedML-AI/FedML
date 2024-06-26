import argparse
import logging

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import Autoscaler
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.computing.scheduler.model_scheduler.autoscaler.policies import ConcurrentQueryPolicy


if __name__ == "__main__":

    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "tester", 0, 0, 0)
    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--redis_addr', default="local")
    parser.add_argument('--redis_port', default=6379)
    parser.add_argument('--redis_password', default="fedml_default")
    args = parser.parse_args()

    fedml_model_cache = FedMLModelCache.get_instance()
    fedml_model_cache.set_redis_params(args.redis_addr, args.redis_port, args.redis_password)

    # Get all endpoints info
    endpoints_settings_list = fedml_model_cache.get_all_endpoints_user_setting()

    # Init the autoscaler
    autoscaler = Autoscaler(args.redis_addr, args.redis_port, args.redis_password)

    autoscaling_policy_config = {
            "current_replicas": 1,
            "min_replicas": 1,
            "max_replicas": 3,
            "queries_per_replica": 2,
            "window_size_secs": 60,
            "scaledown_delay_secs": 120,
    }
    autoscaling_policy = ConcurrentQueryPolicy(**autoscaling_policy_config)

    # Please replace the `e_id` below with a proper e_id value.
    e_id = 1111
    scale_op = autoscaler.scale_operation_endpoint(
        autoscaling_policy,
        str(e_id))
    logging.info(f"Scaling operation {scale_op.value} for endpoint {e_id} .")
