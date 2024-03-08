from fedml.computing.scheduler.model_scheduler.autoscaler.autoscaler import FedMLAutoscaler


def scale_operation_all_endpoints_test():
    autoscaler = FedMLAutoscaler.get_instance()
    autoscaler.scale_operation_all_endpoints()


if __name__ == "__main__":
    scale_operation_all_endpoints_test()
