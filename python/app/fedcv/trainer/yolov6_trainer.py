import logging
import torch
from fedml.core import ClientTrainer

from YOLOv6.yolov6.core.engine import Trainer
from model.util import EnsembleModel

class YOLOv6Trainer(ClientTrainer):
    def __init__(self, model, args, yolo_args, cfg, net_dataidx_map):
        super(YOLOv6Trainer, self).__init__(model, args)
        logging.info('Init YOLOv6Trainer')
        self.args = args
        self.yolo_args = yolo_args
        self.cfg = cfg
        self.net_dataidx_map = net_dataidx_map
        self.round_loss = []
        self.round_idx = 0
        self.meituan_trainer = Trainer(self.yolo_args, self.cfg, device=torch.device('cpu'), data_idx=self.net_dataidx_map[self.id])

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def train(self, train_data, device, args):
        logging.info("Start training on Trainer {}".format(self.id))
        self.meituan_trainer.device = device
        self.meituan_trainer.model.to(device)
        self.meituan_trainer.ema.ema.to(device)
        self.meituan_trainer.model.load_state_dict(self.model.model.state_dict(), strict=True)
        self.meituan_trainer.ema.ema.load_state_dict(self.model.ema.state_dict(), strict=True)
        self.meituan_trainer.ema.updates = self.round_idx
        self.meituan_trainer.ema.decay(self.meituan_trainer.ema.updates)
        self.meituan_trainer.max_epoch = args.epochs
        self.meituan_trainer.max_step = args.steps
        self.meituan_trainer.train()
        enselble_model = EnsembleModel(self.meituan_trainer.model, self.meituan_trainer.ema.ema)
        self.model.load_state_dict(enselble_model.state_dict(), strict=True)

        return 