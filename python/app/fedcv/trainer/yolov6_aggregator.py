import logging
import torch
from fedml.core import ServerAggregator

from YOLOv6.yolov6.core.engine import Trainer

class YOLOv6Aggregator(ServerAggregator):
    def __init__(self, model, args, yolo_args, cfg):
        super(YOLOv6Aggregator, self).__init__(model, args)
        logging.info('Init YOLOv6Aggregator')
        self.args = args
        self.yolo_args = yolo_args
        self.cfg = cfg
        self.round_idx = 0
        self.meituan_trainer = Trainer(self.yolo_args, self.cfg, torch.device('cpu'))

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=True)

    def test(self, test_data, device, args):
        pass

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        if (self.round_idx + 1) % self.args.frequency_of_the_test == 0:
            self.meituan_trainer.device = device
            self.meituan_trainer.model.to(device)
            self.meituan_trainer.ema.ema.to(device)
            self.meituan_trainer.epoch = 0
            self.meituan_trainer.model.load_state_dict(self.model.model.state_dict(), strict=True)
            self.meituan_trainer.ema.ema.load_state_dict(self.model.ema.state_dict(), strict=True)
            self.meituan_trainer.ema.update_attr(self.meituan_trainer.model, include=['nc', 'names', 'stride']) # update attributes for ema model
            self.meituan_trainer.eval_model()

            results = self.meituan_trainer.evaluate_results
            logging.info('Results: {}'.format(results))
            import time
            time.sleep(20)
        self.round_idx += 1
