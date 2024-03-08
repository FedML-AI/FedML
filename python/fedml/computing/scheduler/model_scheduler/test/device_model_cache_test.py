import logging

from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache

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
    # _end_point_id_ = "4f63aa70-312e-4a9c-872d-cc6e8d95f95b"
    # _end_point_name_ = "my-llm"
    # _model_name_ = "my-model"
    # _model_version_ = "v1"
    # _status_list_ = FedMLModelCache.get_instance().get_deployment_status_list(_end_point_id_, _end_point_name_, _model_name_)
    # _result_list_ = FedMLModelCache.get_instance().get_deployment_result_list(_end_point_id_, _end_point_name_, _model_name_)
    # idle_result_payload = FedMLModelCache.get_instance().get_idle_device(_end_point_id_, _end_point_name_, _model_name_, _model_version_)
    get_endpoints_ids_test()
    get_all_endpoints_metrics_test()
    get_all_replicas_test()
