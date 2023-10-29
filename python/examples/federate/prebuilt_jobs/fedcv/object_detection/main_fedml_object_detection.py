import fedml
from fedml import FedMLRunner
from model.init_yolo import init_yolo
from trainer.yolo_aggregator import YOLOAggregator

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # init yolo
    model, dataset, trainer, args = init_yolo(args=args, device=device)
    aggregator = YOLOAggregator(model, args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
