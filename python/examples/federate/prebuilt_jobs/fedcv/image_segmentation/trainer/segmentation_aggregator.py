import logging
import time
import numpy as np
import torch
import torch.nn as nn

import fedml
from fedml.core import ServerAggregator

from fedml.simulation.mpi.fedseg.utils import (
    SegmentationLosses,
    Evaluator,
    LR_Scheduler,
    EvaluationMetricsKeeper,
)


class SegmentationAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device):
        logging.info("Evaluating on Trainer ID: {}".format(self.id))
        model = self.model
        args = self.args
        evaluator = Evaluator(model.n_classes)

        model.eval()
        model.to(device)

        t = time.time()
        evaluator.reset()
        test_acc = (
            test_acc_class
        ) = test_mIoU = test_FWIoU = test_loss = test_total = 0.0
        criterion = SegmentationLosses().build_loss(mode=args.loss_type)

        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_data):
                x, target = batch["image"], batch["label"]
                x, target = x.to(device), target.to(device)
                output = model(x)
                loss = criterion(output, target).to(device)
                test_loss += loss.item()
                test_total += target.size(0)
                pred = output.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                evaluator.add_batch(target, pred)
                if batch_idx % 100 == 0:
                    logging.info(
                        "Trainer_ID: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}".format(
                            self.id, batch_idx, loss, (time.time() - t) / 60
                        )
                    )

                # time_end_test_per_batch = time.time()
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
                # logging.info("Client = {0} Batch = {1}".format(self.client_index, batch_idx)

        # Evaluation Metrics (Averaged over number of samples)
        test_acc = evaluator.Pixel_Accuracy()
        test_acc_class = evaluator.Pixel_Accuracy_Class()
        test_mIoU = evaluator.Mean_Intersection_over_Union()
        test_FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        test_loss = test_loss / test_total

        logging.info(
            "Trainer_ID={0}, test_acc={1}, test_acc_class={2}, test_mIoU={3}, test_FWIoU={4}, test_loss={5}".format(
                self.id, test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss
            )
        )

        eval_metrics = EvaluationMetricsKeeper(
            test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss
        )
        return eval_metrics

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return True
