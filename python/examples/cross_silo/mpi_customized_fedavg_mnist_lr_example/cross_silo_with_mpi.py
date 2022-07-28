import fedml
from my_model_trainer import MyModelTrainer
from my_server_aggregator import MyServerAggregator
from fedml import FedMLRunner

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # my customized trainer and aggregator
    trainer = MyModelTrainer(model, args)
    aggregator = MyServerAggregator(model, args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
