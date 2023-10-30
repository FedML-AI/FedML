import fedml
from fedml import FedMLRunner
from data.data_loader import load_data
from model import create_model
from trainer.segmentation_trainer import SegmentationTrainer
from trainer.segmentation_aggregator import SegmentationAggregator

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, class_num = load_data(args)

    # create model and trainer
    model = create_model(args)
    trainer = SegmentationTrainer(model=model, args=args)
    aggregator = SegmentationAggregator(model=model, args=args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
