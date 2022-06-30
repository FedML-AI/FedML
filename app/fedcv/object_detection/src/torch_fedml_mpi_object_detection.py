import logging
import fedml
from fedml.simulation import SimulatorMPI
from trainer.detection_trainer import DetectionTrainer
from utils.init_yolo import init_yolo


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # init yolo
    model, dataset, args = init_yolo(args=args, device=device)

    # trainer
    trainer = DetectionTrainer(model=model, args=args)

    # start training
    try:
        simulator = SimulatorMPI(args, device, dataset, model, trainer)
        simulator.run()
    except Exception as e:
        raise e
