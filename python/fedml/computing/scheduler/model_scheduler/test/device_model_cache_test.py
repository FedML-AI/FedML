import logging

from collections import namedtuple
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog

logging.getLogger().setLevel(logging.DEBUG)


def get_endpoints_ids_test():
    fedml_model_cache = FedMLModelCache.get_instance()
    fedml_model_cache.set_redis_params()
    endpoint_ids = fedml_model_cache.get_endpoints_ids()
    logging.debug("Endpoint IDs: {}".format(endpoint_ids))


def get_all_endpoints_metrics_test():
    fedml_model_cache = FedMLModelCache.get_instance()
    fedml_model_cache.set_redis_params()
    endpoints_ids = fedml_model_cache.get_endpoints_ids()
    for eid in endpoints_ids:
        # None, -1 and 0 means fetch all the values.
        for k in [None, -1, 0, 1, 3]:
            endpoint_metrics = fedml_model_cache.get_endpoint_metrics(
                endpoint_id=eid, k_recent=k)
            logging.debug("Endpoint ID: {}, Top k-recent: {}, Metrics: {}"
                          .format(eid, k, endpoint_metrics))


def get_all_replicas_test():
    fedml_model_cache = FedMLModelCache.get_instance()
    fedml_model_cache.set_redis_params()
    endpoints_ids = fedml_model_cache.get_endpoints_ids()
    for eid in endpoints_ids:
        endpoint_replicas = fedml_model_cache.get_all_replicas(
            endpoint_id=eid)
        logging.debug("Endpoint ID: {}, Replicas: {}"
                      .format(eid, endpoint_replicas))


if __name__ == "__main__":
    logging_args = namedtuple('LoggingArgs', [
        'log_file_dir', 'client_id', 'client_id_list', 'role', 'rank', 'run_id', 'server_id'])
    args = logging_args("/tmp", 0, [], "server", 0, 0, 0)
    MLOpsRuntimeLog.get_instance(args).init_logs(log_level=logging.DEBUG)
    get_endpoints_ids_test()
    get_all_endpoints_metrics_test()
    get_all_replicas_test()
