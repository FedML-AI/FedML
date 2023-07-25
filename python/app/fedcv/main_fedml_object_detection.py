import logging

import fedml
from fedml import FedMLRunner

from model.init_yolov6 import init_yolov6
from trainer.yolov6_aggregator import YOLOv6Aggregator

if __name__ == "__main__":
    logging.info("Init FedML framework")
    args = fedml.init()
    device = fedml.device.get_device(args)

    logging.info("Init YOLOv6")
    ensemble_model, dataset, trainer, args, yolo_args, yolo_cfg = init_yolov6(args=args, device=device)

    logging.info("Init Aggregator")
    aggregator = YOLOv6Aggregator(ensemble_model, args, yolo_args, yolo_cfg)

    logging.info("Init FedMLRunner")
    fedml_runner = FedMLRunner(args, device, dataset, ensemble_model, trainer, aggregator)

    logging.info("Start training")
    fedml_runner.run()
