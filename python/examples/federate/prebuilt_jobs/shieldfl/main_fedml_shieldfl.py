import fedml
from fedml import FedMLRunner

from data.data_loader import load_shieldfl_data
from model.model_hub import create_model
from trainer.shieldfl_aggregator import ShieldFLAggregator
from trainer.verifl_v16_trainer import VeriFLv16Trainer
from utils.runtime import configure_runtime


if __name__ == "__main__":
    args = fedml.init(check_env=False)
    configure_runtime(args)

    device = fedml.device.get_device(args)
    args.device = device

    dataset, data_assets = load_shieldfl_data(args)
    model = create_model(args)

    trainer = VeriFLv16Trainer(model=model, args=args)

    aggregator = ShieldFLAggregator(model=model, args=args, data_assets=data_assets, device=device)

    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
