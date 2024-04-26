import argparse
import logging

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import Autoscaler, ReactivePolicy
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache


if __name__ == "__main__":

    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "tester", 0, 0, 0)
    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--redis_addr', default="local")
    parser.add_argument('--redis_port', default=6379)
    parser.add_argument('--redis_password', default="fedml_default")
    parser.add_argument('--metric',
                        default="latency",
                        help="Either latency or qps")
    args = parser.parse_args()

    fedml_model_cache = FedMLModelCache.get_instance()
    fedml_model_cache.set_redis_params(args.redis_addr, args.redis_port, args.redis_password)

    # Get all endpoints info
    endpoints_settings_list = fedml_model_cache.get_all_endpoints_user_setting()

    # Init the autoscaler
    autoscaler = Autoscaler(args.redis_addr, args.redis_port, args.redis_password)

    latency_reactive_policy_default = {
        "metric": "latency",
        "ewm_mins": 15,
        "ewm_alpha": 0.5,
        "ub_threshold": 0.5,
        "lb_threshold": 0.99,
        "triggering_value": 1.6561916828471053
    }
    qps_reactive_policy_default = {
        "metric": "qps",
        "ewm_mins": 15,
        "ewm_alpha": 0.5,
        "ub_threshold": 2,
        "lb_threshold": 0.5
    }
    policy_config = latency_reactive_policy_default \
        if args.metric == "latency" else qps_reactive_policy_default
    autoscaling_policy = ReactivePolicy(**policy_config)

    for endpoint_settings in endpoints_settings_list:
        endpoint_state = endpoint_settings["state"]
        if endpoint_state == "DEPLOYED" and endpoint_settings["enable_auto_scaling"]:

            e_id, e_name, model_name = \
                endpoint_settings["endpoint_id"], \
                endpoint_settings["endpoint_name"], \
                endpoint_settings["model_name"]
            logging.info(f"Querying the autoscaler for endpoint {e_id} with user settings {endpoint_settings}.")

            # For every endpoint we just update the policy configuration.
            autoscaling_policy.min_replicas = endpoint_settings["scale_min"]
            autoscaling_policy.max_replicas = endpoint_settings["scale_max"]
            # We retrieve a list of replicas for every endpoint. The number
            # of running replicas is the length of that list.
            current_replicas = len(fedml_model_cache.get_endpoint_replicas_results(e_id))
            autoscaling_policy.current_replicas = current_replicas
            logging.info(f"Endpoint {e_id} autoscaling policy: {autoscaling_policy}.")

            scale_op = autoscaler.scale_operation_endpoint(
                autoscaling_policy,
                str(e_id))

            new_replicas = current_replicas + scale_op.value

            logging.info(f"Scaling operation {scale_op.value} for endpoint {e_id} .")
            logging.info(f"New Replicas {new_replicas} for endpoint {e_id} .")
            logging.info(f"Current Replicas {current_replicas} for endpoint {e_id} .")
