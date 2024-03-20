import logging

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import Autoscaler, ReactivePolicy
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog


def scale_operation_endpoints_reactive_test():
    autoscaler = Autoscaler.get_instance()
    autoscaling_policy = ReactivePolicy(metric="latency")
    autoscaler.scale_operation_endpoints(autoscaling_policy)


if __name__ == "__main__":
    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "server", 0, 0, 0)
    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.DEBUG)
    scale_operation_endpoints_reactive_test()
